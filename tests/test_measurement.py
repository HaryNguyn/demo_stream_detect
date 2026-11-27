"""Pure measurement helpers: resolution scaling, batch summaries, matrix parsing."""
import unittest
from unittest.mock import Mock, patch

import numpy as np

from vehicle_ai.measurement import (
    compact_result, parse_video_name, rebase_config_paths, resolution_overrides,
    summarize_images,
)
from vehicle_ai.pipeline import analyze_video
from vehicle_ai.config import PipelineConfig


class ResolutionMeasurementTests(unittest.TestCase):
    def test_thresholds_scale_with_height_and_keep_minimums(self):
        base = {"min_vehicle_width": 120, "min_plate_width": 32,
                "quality_target_width": 100, "quality_target_height": 32}
        self.assertEqual(resolution_overrides(base, 720), base)
        self.assertEqual(resolution_overrides(base, 144), {
            "min_vehicle_width": 24, "min_plate_width": 8,
            "quality_target_width": 20, "quality_target_height": 8,
        })
        self.assertEqual(resolution_overrides(base, 360), {
            "min_vehicle_width": 60, "min_plate_width": 16,
            "quality_target_width": 50, "quality_target_height": 16,
        })

    def test_only_pixel_thresholds_are_scaled(self):
        base = {"detector_imgsz": 640, "plate_imgsz": 640, "min_vehicle_width": 120}
        overrides = resolution_overrides(base, 240)
        self.assertNotIn("detector_imgsz", overrides)
        self.assertNotIn("plate_imgsz", overrides)

    def test_relative_model_paths_are_rebased(self):
        data = {"vehicle_weights": "../models/yolo11n.pt", "ocr_weights": "C:/abs/ocr.onnx"}
        result = rebase_config_paths(data, "C:/demo/configs", "C:/demo/configs/resolutions")
        self.assertEqual(result["vehicle_weights"], "../../models/yolo11n.pt")
        self.assertEqual(result["ocr_weights"], "C:/abs/ocr.onnx")

    def test_video_name_parsing(self):
        self.assertEqual(parse_video_name("10min_144p.mp4"), {
            "source_id": "10min_144p", "duration_min": 10, "resolution": "144p"})
        self.assertEqual(parse_video_name("30MIN_1080P.mkv")["resolution"], "1080p")
        self.assertIsNone(parse_video_name("dataset1.mp4"))


class ImageSummaryTests(unittest.TestCase):
    def test_compact_result_drops_payloads_but_keeps_labels(self):
        result = {
            "width": 640, "height": 480, "annotated": "base64-big", "elapsed_ms": 12.5,
            "vehicles": [{"plate_text": "30A12345", "plate_score": 0.9, "accepted": True,
                          "plate_image": "base64-big", "brand": "toyota"}],
        }
        compact = compact_result(result)
        self.assertNotIn("annotated", compact)
        self.assertNotIn("plate_image", compact["vehicles"][0])
        self.assertEqual(compact["vehicles"][0]["plate_text"], "30A12345")

    def test_summary_scores_truth_and_ignores_warmup_for_latency(self):
        rows = [
            {"filename": "a.jpg", "elapsed_ms": 100, "elapsed_total_s": 0.1,
             "vehicles": [{"plate_text": "30A12345", "plate_score": 0.9, "accepted": True,
                           "brand": "toyota", "vehicle_type": "Ô tô con"}]},
            {"filename": "b.jpg", "elapsed_ms": 20, "elapsed_total_s": 0.14, "vehicles": []},
            {"filename": "c.jpg", "elapsed_ms": 40, "elapsed_total_s": 0.18,
             "vehicles": [{"plate_text": "30A12346", "plate_score": 0.8, "accepted": True,
                           "brand": "", "vehicle_type": "Xe máy"}]},
            {"filename": "d.jpg", "elapsed_ms": 20, "elapsed_total_s": 0.20, "error": "boom"},
        ]
        truth = {
            "a.jpg": {"plate": "30A12345", "vehicle_type": "Ô tô con", "brand": "Toyota",
                      "tags": ["144p", "car"]},
            "b.jpg": {"plate": None, "tags": ["144p"]},
            "c.jpg": {"plate": "30A12345", "vehicle_type": "Xe máy", "tags": ["720p"]},
        }
        result = summarize_images(rows, truth, warmup=1)
        self.assertEqual((result["images"], result["errors"]), (4, 1))
        self.assertEqual(result["plates_accepted"], 2)
        self.assertAlmostEqual(result["wall_seconds"], 0.10)
        self.assertEqual(result["median_ms"], 30)
        self.assertEqual(result["p95_ms"], 40)
        self.assertEqual(result["groups"]["plate"]["exact_over_total"], "1/2")
        self.assertEqual(result["groups"]["brand"]["accuracy_on_matched"], 1.0)
        self.assertEqual(result["tag_groups"]["144p"]["plate"]["exact_over_total"], "1/1")
        self.assertEqual(result["tag_groups"]["720p"]["plate"]["exact_over_total"], "0/1")

    def test_truth_matches_relative_filename_and_basename(self):
        rows = [{"filename": "144p/img001.jpg", "elapsed_ms": 5,
                 "vehicles": [{"plate_text": "30A12345", "accepted": True}]}]
        truth = {"144p/img001.jpg": {"plate": "30A12345"}}
        self.assertEqual(summarize_images(rows, truth)["groups"]["plate"]["exact_over_total"], "1/1")
        rows[0]["filename"] = "img001.jpg"
        self.assertEqual(summarize_images(rows, {"img001.jpg": {"plate": "30A12345"}})
                         ["groups"]["plate"]["exact_over_total"], "1/1")

    def test_summary_without_truth_skips_accuracy(self):
        result = summarize_images([{"filename": "a.jpg", "elapsed_ms": 5, "vehicles": []}])
        self.assertNotIn("groups", result)
        self.assertEqual(result["truth_images"], 0)


class SourceSizeTests(unittest.TestCase):
    def test_run_stats_report_actual_frame_size(self):
        capture = Mock()
        capture.isOpened.return_value = True
        capture.get.return_value = 25
        capture.read.side_effect = [
            (True, np.zeros((144, 256, 3), np.uint8)), (False, None),
        ]
        session = Mock()
        session.feed.return_value = []
        session.finalize.return_value = []
        stats = {}
        with (
            patch("vehicle_ai.pipeline.YOLO"),
            patch("vehicle_ai.pipeline.RecognitionSession", return_value=session),
            patch("vehicle_ai.pipeline.open_capture", return_value=capture),
        ):
            analyze_video("video.mp4", "test", "model.pt",
                          config=PipelineConfig(vehicle_weights="v"), on_stats=stats.update)
        self.assertEqual(stats["frame_size"], "256x144")


if __name__ == "__main__":
    unittest.main()
