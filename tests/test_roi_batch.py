import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from ultralytics.engine.results import Results

from vehicle_ai.config import PipelineConfig
from vehicle_ai.inference import locate_plate
from vehicle_ai.ocr import VietnamesePlateRecognizer
from vehicle_ai.session import RecognitionSession


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
