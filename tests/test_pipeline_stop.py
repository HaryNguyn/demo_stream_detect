"""Cancellation must finalize tracked vehicles and release the source."""
import unittest
from unittest.mock import Mock, patch

from vehicle_ai.pipeline import analyze_video


class PipelineStopTests(unittest.TestCase):
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
