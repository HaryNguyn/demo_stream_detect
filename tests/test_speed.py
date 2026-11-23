"""Speed tuning: presets, evidence encoding and detector frame stride."""
import unittest
from unittest.mock import Mock, patch

import numpy as np

from vehicle_ai.aggregation import Vehicle
from vehicle_ai.config import PipelineConfig, SPEED_PRESETS, apply_speed_preset
from vehicle_ai.inference import add_vehicle_sample
from vehicle_ai.pipeline import analyze_video
from vehicle_ai.web_models import prepare_models


class SpeedPresetTests(unittest.TestCase):
    def test_presets_build_valid_configs_and_override_tuning(self):
        for name, expected in SPEED_PRESETS.items():
            with self.subTest(preset=name):
                config = apply_speed_preset(PipelineConfig(vehicle_weights="v"), name)
                for field, value in expected.items():
                    self.assertEqual(getattr(config, field), value)

    def test_custom_preset_keeps_configuration(self):
        config = PipelineConfig(vehicle_weights="v", detector_imgsz=640, ocr_top_k=4)
        self.assertIs(apply_speed_preset(config, "custom"), config)
        self.assertEqual((config.detector_imgsz, config.ocr_top_k), (640, 4))

    def test_invalid_stride_and_preview_rate_are_rejected(self):
        with self.assertRaises(ValueError):
            PipelineConfig(vehicle_weights="v", detector_frame_stride=0)
        with self.assertRaises(ValueError):
            PipelineConfig(vehicle_weights="v", preview_fps=0)
        with self.assertRaises(ValueError):
            PipelineConfig(vehicle_weights="v", preview_fps=120)

    def test_ui_speed_preset_is_applied_to_models(self):
        with (
            patch("vehicle_ai.web_models.load_config", return_value=PipelineConfig(vehicle_weights="v")),
            patch("vehicle_ai.inference.optional_yolo", return_value=None),
            patch("vehicle_ai.model_loader.load_recognizer", return_value=None),
        ):
            config, _, _, _ = prepare_models({"mode": "yaml", "speed_preset": "fast"})
        self.assertEqual(config.detector_imgsz, 416)
        self.assertEqual(config.detector_frame_stride, 2)
        self.assertEqual(config.preview_fps, 12)


class EvidenceEncodingTests(unittest.TestCase):
    def test_full_evidence_list_skips_worse_encodes(self):
        vehicle = Vehicle(0, candidate_limit=2)
        image = np.full((40, 80, 3), 90, np.uint8)
        with patch("vehicle_ai.inference.encode_crop", side_effect=lambda item: {"ok": True}) as encoder:
            for frame_idx, score in enumerate((0.9, 0.8, 0.1)):
                add_vehicle_sample(vehicle, image, score, 3200, 6400, frame_idx)
            self.assertEqual(encoder.call_count, 2)
            self.assertEqual(len(vehicle.detected_samples), 2)
            add_vehicle_sample(vehicle, image, 0.95, 3200, 6400, 3)
            self.assertEqual(encoder.call_count, 3)

    def test_keeps_sample_until_limit(self):
        vehicle = Vehicle(0, candidate_limit=3)
        self.assertTrue(vehicle.keeps_sample("plate", 0.1))
        vehicle.plate_samples = [{"_score": 0.6}, {"_score": 0.7}, {"_score": 0.8}]
        self.assertFalse(vehicle.keeps_sample("plate", 0.55))
        self.assertTrue(vehicle.keeps_sample("plate", 0.85))


class DetectorStrideTests(unittest.TestCase):
    def test_stride_skips_detector_feeds_but_keeps_timestamps_increasing(self):
        capture = Mock()
        capture.isOpened.return_value = True
        capture.get.return_value = 10
        frames = [(True, object()) for _ in range(7)] + [(False, None)]
        capture.read.side_effect = frames
        session = Mock()
        session.feed.return_value = []
        session.finalize.return_value = []
        with (
            patch("vehicle_ai.pipeline.YOLO"),
            patch("vehicle_ai.pipeline.RecognitionSession", return_value=session),
            patch("vehicle_ai.pipeline.open_capture", return_value=capture),
        ):
            analyze_video("video.mp4", "test", "model.pt",
                          config=PipelineConfig(vehicle_weights="v", detector_frame_stride=2))
        fed = [call.args[1] for call in session.feed.call_args_list]
        self.assertEqual(fed, [0.0, 0.2, 0.4, 0.6])
        self.assertEqual(session.feed.call_count, 4)


if __name__ == "__main__":
    unittest.main()
