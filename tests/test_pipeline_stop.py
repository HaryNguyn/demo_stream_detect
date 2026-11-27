"""Cancellation must finalize tracked vehicles and release the source."""
import unittest
from unittest.mock import Mock, patch

from vehicle_ai.config import PipelineConfig
from vehicle_ai.pipeline import analyze_video


class PipelineStopTests(unittest.TestCase):
    def test_realtime_duration_uses_source_frames_skipped_by_stride(self):
        capture = Mock()
        capture.isOpened.return_value = True
        capture.get.return_value = 25
        capture.read.return_value = (True, object())
        session = Mock()
        session.feed.return_value = []
        session.finalize.return_value = []
        stats = {}
        with (
            patch("vehicle_ai.pipeline.YOLO"),
            patch("vehicle_ai.pipeline.RecognitionSession", return_value=session),
            patch("vehicle_ai.pipeline.open_capture", return_value=capture),
        ):
            session.tracks = {}
            analyze_video(
                "video.mp4", "test", "model.pt", realtime=True, max_seconds=0.08,
                config=PipelineConfig(vehicle_weights="model.pt", detector_frame_stride=3),
                on_stats=stats.update,
            )
        session.feed.assert_called_once_with(capture.read.return_value[1], 0)
        session.finalize.assert_called_once_with("duration_limit")
        self.assertEqual(stats["frames"], 2)
        self.assertEqual(stats["analyzed_frames"], 1)
        self.assertEqual(stats["content_seconds"], 0.08)
        self.assertEqual(stats["unprocessed_eligible_frames"], 0)

    def test_stop_finalizes_and_releases(self):
        capture = Mock()
        capture.isOpened.return_value = True
        capture.get.return_value = 25
        capture.read.return_value = (True, object())
        session = Mock()
        session.feed.return_value = []
        session.finalize.return_value = [{"Track ID": 1}]
        callback = Mock()
        with (
            patch("vehicle_ai.pipeline.YOLO"),
            patch("vehicle_ai.pipeline.RecognitionSession", return_value=session),
            patch("vehicle_ai.pipeline.open_capture", return_value=capture),
        ):
            rows = analyze_video(
                "video.mp4", "test", "model.pt", on_event=callback,
                should_stop=Mock(side_effect=[False, False, True]),
            )
        self.assertEqual(rows, [{"Track ID": 1}])
        session.finalize.assert_called_once_with("user_stop")
        callback.assert_called_once_with({"Track ID": 1})
        capture.release.assert_called_once()
        capture.read.assert_called_once()

    def test_stop_before_loading(self):
        with patch("vehicle_ai.pipeline.YOLO") as detector:
            self.assertEqual(analyze_video("v", "c", "m", should_stop=lambda: True), [])
            detector.assert_not_called()
