"""RTSP URL construction and camera-page validation."""
import unittest

from vehicle_ai.capture import open_capture, stream_source
from vehicle_ai.ip_camera import build_rtsp_url, looks_like_web_page, mask_url, validate_stream_url


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


if __name__ == "__main__":
    unittest.main()
