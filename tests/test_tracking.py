"""Vehicle tracking, ROI, recognition budgets, and batched OCR."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import cv2
import numpy as np
import torch
from ultralytics.engine.results import Results

from vehicle_ai.aggregation import Vehicle
from vehicle_ai.config import PipelineConfig
from vehicle_ai.inference import draw_anpr_preview, draw_recognition, locate_plate, recognize_vehicle
from vehicle_ai.ocr import VietnamesePlateRecognizer
from vehicle_ai.roi import center_in_roi, validate_roi
from vehicle_ai.session import RecognitionSession
from vehicle_ai.storage import ExperimentWriter


class SessionTests(unittest.TestCase):
    def make_session(self):
        frame = np.full((80, 120, 3), 127, np.uint8)
        found = Results(frame, path="", names={0: "car"},
                        boxes=torch.tensor([[5, 5, 100, 70, 1, 0.9, 0]]))
        empty = Results(frame, path="", names={0: "car"}, boxes=torch.empty((0, 7)))
        detector = MagicMock()
        detector.track.side_effect = [[found], [empty]]
        # Small synthetic crop: exercise evidence storage without the production size gate.
        config = PipelineConfig(vehicle_weights="unused", vehicle_classes={0: "car"}, min_vehicle_width=0)
        return RecognitionSession(detector, config, "cam", source_id="video"), frame

    def test_timeout_outputs_event_once_and_saves_evidence(self):
        session, frame = self.make_session()
        self.assertEqual(session.feed(frame, 0), [])
        events = session.feed(frame, 3)
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event["event"]["status"], "no_plate")
        self.assertEqual(event["event"]["vehicle_type"], "car")
        self.assertEqual(event["event"]["finalization_reason"], "track_timeout")
        self.assertEqual(event["event"]["ended_at_seconds"], 0)
        self.assertEqual(session.finalize(), [])
        with self.assertRaises(RuntimeError):
            session.feed(frame, 4)
        with tempfile.TemporaryDirectory() as tmp:
            writer = ExperimentWriter(Path(tmp) / "run")
            writer.append(event)
            writer.close()
            saved = json.loads((Path(tmp) / "run" / "events.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(len(saved["evidence"]), 1)
            self.assertIsNotNone(cv2.imread(str(Path(tmp) / "run" / saved["evidence"][0]["path"])))
            self.assertNotIn("base64", json.dumps(saved))

    def test_finalize_flushes_and_is_idempotent(self):
        session, frame = self.make_session()
        session.feed(frame, 1)
        with self.assertRaises(ValueError):
            session.feed(frame, 0.5)
        with self.assertRaises(ValueError):
            session.feed(frame, float("nan"))
        self.assertEqual(len(session.finalize()), 1)
        self.assertEqual(session.finalize(), [])


class ROITests(unittest.TestCase):
    def test_anpr_draws_plate_box_in_frame_coordinates(self):
        frame = np.full((480, 640, 3), 77, np.uint8)
        result = Results(frame, path="", names={0: "car"}, boxes=torch.tensor([
            [200, 200, 400, 400, 1, .9, 0],
        ]))
        vehicle = Vehicle(0, plate_box=(.25, .6, .75, .8),
                          plate_detection_score=.9, plate_ocr_score=.95)
        vehicle.add_plate("29A12345", .95)
        preview = draw_anpr_preview(frame, result, {1: vehicle}, inference_fps=12)
        # Plate at (250,320)-(350,360), rather than the entire vehicle.
        self.assertFalse(np.array_equal(preview[360, 300], frame[360, 300]))
        np.testing.assert_array_equal(preview[400, 300], frame[400, 300])
        self.assertEqual(preview.shape, frame.shape)
        self.assertTrue(np.all(frame == 77))
        outside = [[0, 0], [.2, 0], [.2, 1], [0, 1]]
        filtered = draw_anpr_preview(frame, result, {1: vehicle}, outside)
        np.testing.assert_array_equal(filtered[360, 300], frame[360, 300])

    def test_live_overlay_uses_current_video_frame(self):
        old = np.zeros((480, 640, 3), np.uint8)
        current = np.full_like(old, 77)
        result = Results(old, path="", names={0: "car"}, boxes=torch.tensor([
            [200, 200, 300, 300, 1, .9, 0],
        ]))
        roi = [[.25, .25], [.75, .25], [.75, .75], [.25, .75]]
        preview = draw_recognition(result, {}, roi, frame=current)
        np.testing.assert_array_equal(preview[0, 0], current[0, 0])
        self.assertFalse(np.array_equal(preview[300, 250], current[300, 250]))
        self.assertFalse(np.array_equal(preview[240, 160], current[240, 160]))
        self.assertTrue(np.all(current == 77))
        self.assertTrue(np.all(old == 0))

    def test_normalized_roi_works_for_multiple_resolutions(self):
        roi = validate_roi([[.25, .25], [.75, .25], [.75, .75], [.25, .75]])
        for scale in (1, 2, 4):
            shape = (100 * scale, 100 * scale, 3)
            self.assertTrue(center_in_roi([40*scale, 40*scale, 60*scale, 60*scale], shape, roi))
            self.assertFalse(center_in_roi([0, 0, 10*scale, 10*scale], shape, roi))
        self.assertTrue(center_in_roi([0, 0, 1, 1], (100, 100, 3), []))

    def test_invalid_polygons(self):
        for polygon in (
            [[0, 0], [1, 1]], [[0, 0], [.5, .5], [1, 1]],
            [[0, 0], [2, 0], [1, 1]], [[0, 0], [float("nan"), 0], [1, 1]],
            [[0, 0], [1, 1], [0, 1], [1, 0]],
        ):
            with self.subTest(polygon=polygon), self.assertRaises(ValueError):
                validate_roi(polygon)

    def test_session_ignores_vehicle_outside_roi(self):
        image = np.zeros((100, 100, 3), np.uint8)
        result = Results(image, path="", names={0: "car"}, boxes=torch.tensor([
            [40, 40, 60, 60, 1, .9, 0],
            [0, 0, 10, 10, 2, .9, 0],
        ]))
        detector = MagicMock()
        detector.track.return_value = [result]
        config = PipelineConfig(vehicle_weights="unused", vehicle_classes={0: "car"},
                                roi=[[.25, .25], [.75, .25], [.75, .75], [.25, .75]],
                                crop_vehicle_roi=False)
        session = RecognitionSession(detector, config, "cam")
        session.feed(image, 0)
        preview = draw_recognition(result, session.tracks, config.roi)
        self.assertEqual(preview.shape, image.shape)
        events = session.finalize()
        self.assertEqual([row["Track ID"] for row in events], [1])
        config.roi = []
        session = RecognitionSession(detector, config, "cam")
        session.feed(image, 0)
        self.assertEqual(len(session.finalize()), 2)


class ROIBatchTests(unittest.TestCase):
    def test_vehicle_crop_restores_coordinates_and_keeps_polygon_filter(self):
        frame = np.zeros((400, 800, 3), np.uint8)
        detector = Mock()

        def track(cropped, **kwargs):
            self.assertEqual(cropped.shape[:2], (200, 400))
            return [Results(cropped, path="", names={2: "car"}, boxes=torch.tensor([
                [80, 70, 160, 130, 1, .9, 2],
                [350, 0, 390, 20, 2, .9, 2],
            ]))]

        detector.track.side_effect = track
        config = PipelineConfig(vehicle_weights="unused", vehicle_roi_padding=0,
                                roi=[[.25, .25], [.75, .75], [.25, .75]])
        session = RecognitionSession(detector, config, "cam")
        session.feed(frame, 0)
        self.assertEqual(set(session.tracks), {1})
        np.testing.assert_allclose(session.last_result.boxes.xyxy[0], [280, 170, 360, 230])
        self.assertIs(session.last_result.orig_img, frame)

    def test_plate_subroi_offsets_and_full_car_fallback(self):
        image = np.zeros((200, 300, 3), np.uint8)
        model = Mock()
        empty = Results(image, path="", names={0: "plate"}, boxes=torch.empty((0, 6)))
        found = Results(image, path="", names={0: "plate"},
                        boxes=torch.tensor([[10, 20, 90, 40, .9, 0]]))
        model.predict.side_effect = [[found]]
        coords, _ = locate_plate(model, image, lower_fraction=.5)
        self.assertEqual(coords, (10, 120, 90, 140))
        self.assertEqual(model.predict.call_args.args[0].shape[:2], (100, 300))
        model.predict.reset_mock()
        model.predict.side_effect = [[empty], [found]]
        coords, _ = locate_plate(model, image, lower_fraction=.5)
        self.assertEqual(coords, (10, 20, 90, 40))
        self.assertEqual(model.predict.call_count, 2)
        self.assertEqual(model.predict.call_args.args[0].shape, image.shape)

    def test_ocr_batch_preserves_order_and_pads_static_batch(self):
        for fixed in (None, 1, 4):
            recognizer = VietnamesePlateRecognizer.__new__(VietnamesePlateRecognizer)
            recognizer.height, recognizer.width = 64, 128
            recognizer.alphabet, recognizer.pad_char = "01_", "_"
            recognizer.input_name = "input"
            recognizer.fixed_batch = fixed
            recognizer.batch_size = fixed or 8
            recognizer.session = Mock()

            def run(outputs, feed):
                batch = feed["input"]
                self.assertEqual(batch.dtype, np.uint8)
                self.assertEqual(batch.shape[1:], (64, 128, 3))
                if fixed:
                    self.assertEqual(len(batch), fixed)
                probabilities = np.zeros((len(batch), 5, 3), np.float32)
                for i, item in enumerate(batch):
                    probabilities[i, :, int(item[0, 0, 0])] = 1
                return [probabilities]

            recognizer.session.run.side_effect = run
            images = [np.full((24, 80, 3), value, np.uint8) for value in (0, 1, 0)]
            rows = recognizer.predict_batch(images)
            self.assertEqual([r[0]["rec_text"] for r in rows], ["00000", "11111", "00000"])
            self.assertEqual(recognizer.session.run.call_count, 3 if fixed == 1 else 1)

    def test_session_gathers_multiple_tracks_into_one_ocr_batch(self):
        frame = np.zeros((400, 800, 3), np.uint8)
        result = Results(frame, path="", names={2: "car"}, boxes=torch.tensor([
            [0, 0, 200, 200, 1, .9, 2], [300, 0, 500, 200, 2, .9, 2],
        ]))
        detector = Mock()
        detector.track.return_value = [result]
        session = RecognitionSession(detector, PipelineConfig(vehicle_weights="unused"), "cam",
                                     recognizer=Mock())

        def prepare(vehicle, image, *args, pending_ocr, **kwargs):
            diagnostic = {"status": "Chờ OCR batch"}
            pending_ocr.append((vehicle, image, diagnostic))
            return diagnostic

        reads = [{"text": text, "score": .95, "accepted": True} for text in ("29A12345", "30A67890")]
        with patch("vehicle_ai.session.recognize_vehicle", side_effect=prepare), \
                patch("vehicle_ai.session.inspect_plates", return_value=reads) as batch:
            session.feed(frame, 0)
            self.assertEqual(batch.call_count, 1)
            self.assertEqual(len(batch.call_args.args[1]), 2)
        self.assertIn("29A12345", session.tracks[1].plates)
        self.assertIn("30A67890", session.tracks[2].plates)
        self.assertEqual(session.diagnostics["accepted_reads"], 2)


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


if __name__ == "__main__":
    unittest.main()
