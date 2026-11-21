"""Preview pacing and inference backpressure are independent."""
import unittest
from threading import Event
from time import monotonic
from unittest.mock import Mock
from unittest.mock import patch

from vehicle_ai.playback import realtime_frames


class PlaybackTests(unittest.TestCase):
    def test_connection_startup_delay_does_not_discard_first_frame(self):
        capture = Mock()
        clock = [0.0]

        def read():
            clock[0] = 10.0
            return True, "first"

        capture.read.side_effect = read
        preview = Mock()
        with patch("vehicle_ai.playback.monotonic", side_effect=lambda: clock[0]):
            rows = list(realtime_frames(capture, 25, 1, .04, preview))
        self.assertEqual(rows, [(0, "first")])
        preview.assert_called_once_with(1, 1, "first")

    def test_preview_continues_while_inference_is_busy(self):
        capture = Mock()
        capture.read.side_effect = [(True, i) for i in range(6)] + [(False, None)]
        preview_done = Event()
        displayed = []
        start = monotonic()

        def preview(index, total, frame):
            displayed.append((index, monotonic() - start))
            if index == 6:
                preview_done.set()

        frames = realtime_frames(capture, 20, 6, 0, preview)
        try:
            first = next(frames)
            self.assertEqual(first, (0, 0))
            # Simulate blocked AI without blocking the preview producer.
            self.assertTrue(preview_done.wait(2))
            remaining = list(frames)
        finally:
            frames.close()
        self.assertEqual(remaining, [(5, 5)])
        self.assertEqual([index for index, _ in displayed], list(range(1, 7)))
        for index, elapsed in displayed:
            self.assertGreaterEqual(elapsed + 0.005, (index - 1) / 20)

    def test_duration_limit_and_capture_errors(self):
        capture = Mock()
        capture.read.return_value = (True, object())
        rows = list(realtime_frames(capture, 20, 100, 0.1, None))
        self.assertEqual([index for index, _ in rows], [0, 1])
        capture.read.side_effect = ValueError("decode failed")
        with self.assertRaisesRegex(ValueError, "decode failed"):
            list(realtime_frames(capture, 20, 0, 0, None))
