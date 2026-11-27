"""Video capture, IP cameras, playback, cancellation, and speed settings."""

import io
import unittest
from threading import Event
from time import monotonic
from unittest.mock import MagicMock, Mock, patch

import numpy as np

from vehicle_ai import capture as sources
from vehicle_ai.aggregation import Vehicle
from vehicle_ai.capture import open_capture, stream_source
from vehicle_ai.config import PipelineConfig, SPEED_PRESETS, apply_speed_preset
from vehicle_ai.inference import add_vehicle_sample
from vehicle_ai.ip_camera import build_rtsp_url, looks_like_web_page, mask_url, validate_stream_url
from vehicle_ai.pipeline import analyze_video
from vehicle_ai.playback import PlaybackProgress, realtime_frames
from vehicle_ai.web_models import prepare_models


class IpCameraUrlTests(unittest.TestCase):
    def test_hikvision_main_and_sub_streams(self):
        self.assertEqual(
            build_rtsp_url("10.248.166.48", "admin", "secret"),
            "rtsp://admin:secret@10.248.166.48:554/Streaming/Channels/101",
        )
        self.assertEqual(
            build_rtsp_url("10.248.166.48", port=8554, channel=2, stream="sub"),
            "rtsp://10.248.166.48:8554/Streaming/Channels/202",
        )

    def test_dahua_and_custom_paths(self):
        self.assertEqual(
            build_rtsp_url("192.168.1.108", username="admin", password="p",
                          brand="dahua", channel=3, stream="sub"),
            "rtsp://admin:p@192.168.1.108:554/cam/realmonitor?channel=3&subtype=1",
        )
        self.assertEqual(
            build_rtsp_url("camera.local", brand="custom", path="live/main"),
            "rtsp://camera.local:554/live/main",
        )

    def test_credentials_are_percent_encoded(self):
        url = build_rtsp_url("camera.local", username="admin@site", password="p@ss:word")
        self.assertIn("admin%40site:p%40ss%3Aword@camera.local", url)

    def test_invalid_settings_are_rejected(self):
        for kwargs in (
            {"host": ""}, {"host": "camera.local", "port": 0},
            {"host": "camera.local", "channel": 0}, {"host": "camera.local", "stream": "third"},
            {"host": "camera.local", "brand": "unknown"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                build_rtsp_url(**kwargs)

    def test_camera_web_pages_are_rejected_but_video_urls_pass(self):
        page = "http://10.248.166.48/doc/page/preview.asp"
        self.assertTrue(looks_like_web_page(page))
        with self.assertRaisesRegex(ValueError, "trang quản trị"):
            validate_stream_url(page)
        self.assertEqual(validate_stream_url("rtsp://camera.local/Streaming/Channels/101"),
                         "rtsp://camera.local/Streaming/Channels/101")
        self.assertEqual(validate_stream_url("https://example.test/live.m3u8"),
                         "https://example.test/live.m3u8")
        self.assertEqual(
            validate_stream_url("http://camera.local/ISAPI/Streaming/channels/101/httpPreview"),
            "http://camera.local/ISAPI/Streaming/channels/101/httpPreview",
        )
        with self.assertRaises(ValueError):
            validate_stream_url("http://camera.local/")
        with self.assertRaises(ValueError):
            validate_stream_url("ftp://camera.local/live")

    def test_password_is_masked_and_capture_rejects_web_page(self):
        self.assertEqual(mask_url("rtsp://admin:secret@camera.local:554/live"),
                         "rtsp://admin:***@camera.local:554/live")
        label, source = stream_source("rtsp://admin:secret@camera.local:554/live")
        self.assertEqual(source, "rtsp://admin:secret@camera.local:554/live")
        self.assertNotIn("secret", label)
        with self.assertRaisesRegex(ValueError, "trang quản trị"):
            open_capture("http://10.248.166.48/doc/page/preview.asp")


class SourceTests(unittest.TestCase):
    def make_process(self):
        process = MagicMock()
        process.stderr = io.BytesIO(b"diagnostic")
        process.stdout = io.BytesIO()
        process.poll.return_value = 0
        return process

    def test_http_options_do_not_leak_to_rtsp_or_file(self):
        for url, http in [("https://example.test/live", True),
                          ("rtsp://example.test/live", False), ("video.mp4", False)]:
            with self.subTest(url=url):
                with patch.object(sources.shutil, "which", return_value="ffmpeg"):
                    with patch.object(sources, "corporate_proxy", return_value="http://proxy:8080"):
                        with patch.object(sources.subprocess, "Popen", return_value=self.make_process()) as popen:
                            capture = sources.FfmpegStreamCapture(url)
                            capture.release()
                command = popen.call_args.args[0]
                self.assertEqual("-reconnect" in command, http)
                self.assertEqual("-http_proxy" in command, http)
                self.assertEqual(capture.error_text(), "diagnostic")

    def test_stderr_is_drained_with_bounded_memory(self):
        process = MagicMock(stderr=io.BytesIO(b"x" * 200000))
        errors = sources.ProcessErrors(process)
        errors.finish()
        self.assertFalse(errors.thread.is_alive())
        self.assertLessEqual(len(errors.text()), 64 * 1024)


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

        progress = PlaybackProgress()
        frames = realtime_frames(capture, 20, 6, 0, preview, frame_stride=2, progress=progress)
        try:
            first = next(frames)
            self.assertEqual(first, (0, 0))
            # Simulate blocked AI without blocking the preview producer.
            self.assertTrue(preview_done.wait(2))
            remaining = list(frames)
        finally:
            frames.close()
        self.assertEqual(remaining, [(4, 4)])
        self.assertEqual(progress.frames_read, 6)
        self.assertEqual(progress.frames_eligible, 3)
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
