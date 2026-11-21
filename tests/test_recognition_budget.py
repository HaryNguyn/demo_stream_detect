import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from ultralytics.engine.results import Results

from vehicle_ai.aggregation import Vehicle
from vehicle_ai.config import PipelineConfig
from vehicle_ai.inference import recognize_vehicle
from vehicle_ai.session import RecognitionSession


class RecognitionBudgetTests(unittest.TestCase):
    def session(self, boxes, **options):
        frame = np.zeros((600, 800, 3), np.uint8)
        result = Results(frame, path="", names={2: "car"}, boxes=torch.tensor(boxes))
        detector = Mock()
        detector.track.return_value = [result]
        config = PipelineConfig(vehicle_weights="unused", **options)
        return RecognitionSession(detector, config, "cam"), frame

    def test_budget_prioritizes_large_crops_without_starving_other_tracks(self):
        session, frame = self.session([
            [0, 0, 80, 80, 1, .9, 2],
            [0, 0, 160, 160, 2, .9, 2],
            [200, 200, 500, 500, 3, .9, 2],
        ], max_recognitions_per_frame=1)
        with patch("vehicle_ai.session.recognize_vehicle", return_value={"status": "waiting"}) as read:
            session.feed(frame, 0)
            self.assertIs(read.call_args.args[0], session.tracks[3])
            session.feed(frame, .5)
            self.assertIs(read.call_args.args[0], session.tracks[2])
            self.assertEqual(read.call_count, 2)
        self.assertEqual(session.seen_at, {1: .5, 2: .5, 3: .5})
        self.assertEqual(session.detector.track.call_args.kwargs["imgsz"], 640)

    def test_confirmed_plate_stops_recognition_but_keeps_tracking(self):
        session, frame = self.session([[0, 0, 200, 200, 1, .9, 2]])

        def read(vehicle, *args, **kwargs):
            vehicle.add_plate("29A12345", .95)
            vehicle.add_sample("plate", {
                "_score": .8, "_frame_idx": session.frame_idx, "quality_score": .8,
                "detection_score": .9, "ocr_score": .95, "recognized_plate": "29A12345",
            })
            return {"status": "Đọc được biển"}

        with patch("vehicle_ai.session.recognize_vehicle", side_effect=read) as recognize:
            for i in range(8):
                session.feed(frame, i * .5)
            self.assertEqual(recognize.call_count, 4)
        self.assertEqual(session.tracks[1].confirmed_plate, "29A12345")
        self.assertEqual(session.seen_at[1], 3.5)
        self.assertEqual(session.finalize()[0]["Biển số"], "29A12345")

    def test_conflicting_or_low_confidence_votes_do_not_confirm(self):
        vehicle = Vehicle(0)
        for text in ("29A12345", "29A12346"):
            for _ in range(2):
                vehicle.add_plate(text, .95)
        vehicle.confirm_plate(3, 3, .85)
        self.assertEqual(vehicle.confirmed_plate, "")
        for _ in range(10):
            vehicle.add_plate("29A12345", .5)
        vehicle.confirm_plate(3, 3, .85)
        self.assertEqual(vehicle.confirmed_plate, "")

    def test_small_plate_is_detected_but_does_not_run_ocr(self):
        image = np.zeros((200, 200, 3), np.uint8)
        result = Results(image, path="", names={0: "plate"},
                         boxes=torch.tensor([[10, 10, 30, 20, .9, 0]]))
        model = Mock()
        model.predict.return_value = [result]
        vehicle = Vehicle(0)
        with patch("vehicle_ai.inference.inspect_plate") as ocr:
            recognize_vehicle(vehicle, image, None, model, Mock(), min_plate_width=32)
            ocr.assert_not_called()
        self.assertIsNotNone(vehicle.plate_box)
        self.assertEqual(vehicle.plates, {})

    def test_invalid_budget_configuration(self):
        for options in ({"max_recognitions_per_frame": 0}, {"min_vehicle_width": -1},
                        {"plate_confirm_score": float("nan")}, {"detector_half": True},
                        {"plate_confirm_reads": 1.5}, {"plate_confirm_mean": 0}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                PipelineConfig(vehicle_weights="unused", **options)
