"""HTTP contract checks for the local browser UI."""
import csv
from io import BytesIO, StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from event_store_stub import EventStoreStub
from vehicle_ai.web import create_app
from vehicle_ai.config import PipelineConfig


class LocalWebTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.store = EventStoreStub()
        self.client = create_app(self.store).test_client()

    def test_page_and_empty_status(self):
        for url in ("/", "/static/app.js"):
            with self.client.get(url) as response:
                self.assertEqual(response.status_code, 200)
        self.assertEqual(self.client.get("/api/status").json["status"], "idle")
        self.assertEqual(self.client.get("/api/events").json, [])
        self.assertEqual(self.client.get("/api/frame").status_code, 404)

    def test_invalid_input_does_not_start_job(self):
        for data in ({}, {"url": "rtsp://example", "max_seconds": "nan"},
                     {"url": "rtsp://example", "max_seconds": "-1"}):
            self.assertEqual(self.client.post("/api/analyze", data=data).status_code, 400)
        self.assertEqual(self.client.get("/api/status").json["status"], "idle")

    def test_upload_and_busy_response(self):
        with patch("vehicle_ai.web.Thread") as thread:
            response = self.client.post("/api/analyze", data={
                "video": (BytesIO(b"video"), "sample.mp4"),
            })
            self.assertEqual(response.status_code, 202)
            self.assertTrue(thread.return_value.start.called)
            temporary = thread.call_args.kwargs["args"][-1]
            self.addCleanup(temporary.cleanup)
            self.assertEqual((Path(temporary.name) / "video1.mp4").read_bytes(), b"video")
            self.assertEqual(self.client.post("/api/analyze", data={"url": "rtsp://test"}).status_code, 409)
            self.assertEqual(self.client.post("/api/stop").json["status"], "stopping")
            self.assertEqual(self.client.post("/api/stop").json["status"], "stopping")
            self.assertEqual(self.client.post("/api/analyze", data={"url": "rtsp://test"}).status_code, 409)

    def test_stop_when_idle(self):
        self.assertEqual(self.client.post("/api/stop").json["status"], "idle")

    def test_multiple_sources_and_model_settings(self):
        with patch("vehicle_ai.web.Thread") as thread:
            response = self.client.post("/api/analyze", data={
                "video": [(BytesIO(b"a"), "a.mp4"), (BytesIO(b"b"), "b.mov")],
                "url": "https://example.test/live", "mode": "manual",
                "roi_path": "roi.json", "vehicle_weights": "custom.pt",
            })
            self.assertEqual(response.status_code, 202)
            sources, options, _, temporary = thread.call_args.kwargs["args"]
            self.addCleanup(temporary.cleanup)
            self.assertEqual(len(sources), 3)
            self.assertEqual([source[2] for source in sources], [False, False, True])
            self.assertEqual(options["vehicle_weights"], "custom.pt")
            self.assertEqual(options["roi_path"], "roi.json")

    def test_csv_download(self):
        self.store.append({"Camera": "Camera 1", "Track ID": 2, "Biển số": "29A12345"})
        with self.client.get("/api/events.csv") as response:
            self.assertEqual(response.status_code, 200)
            self.assertIn("29A12345", response.data.decode("utf-8-sig"))
            self.assertIn("attachment", response.headers["Content-Disposition"])

    def test_rejection_reasons_are_exposed_in_events_and_csv(self):
        self.store.append({
            "Camera": "test", "Track ID": 1, "Biển số": "29A12345",
            "rejection_reasons": {"merged_duplicate": 1},
            "rejection_detail": {"merged_duplicate": "gộp vào cũ"},
        })
        row = self.client.get("/api/events").json[0]
        self.assertEqual(row["rejection_reasons"], {"merged_duplicate": 1})
        self.assertEqual(row["rejection_detail"]["merged_duplicate"], "gộp vào cũ")
        with self.client.get("/api/events.csv") as response:
            text = response.data.decode("utf-8-sig")
        self.assertIn("rejection_reasons", text.splitlines()[0])
        self.assertIn("merged_duplicate", text)

    def test_diagnostic_exposes_rejection_reasons(self):
        with patch("vehicle_ai.web.Thread") as thread:
            self.client.post("/api/analyze", data={"url": "rtsp://test"})
        run = thread.call_args.kwargs["target"]

        def analyze(*args, **kwargs):
            kwargs["on_diagnostic"]({
                "status": "Không thấy xe trong ROI/khung hình",
                "rejections": {"vehicle_too_small": 3},
                "thresholds": {"min_vehicle_width": 120},
                "frame_size": "640x360",
                "hidden_by_recognized_only": 1,
            })

        with (
            patch("vehicle_ai.web_models.prepare_models", return_value=(PipelineConfig(vehicle_weights="v"), None, None, None)),
            patch("vehicle_ai.capture.stream_source", return_value=("test", "test")),
            patch("vehicle_ai.pipeline.analyze_video", side_effect=analyze),
        ):
            run(*thread.call_args.kwargs["args"])
        diagnostic = self.client.get("/api/status").json["diagnostic"]
        self.assertEqual(diagnostic["rejections"], {"vehicle_too_small": 3})
        self.assertEqual(diagnostic["thresholds"]["min_vehicle_width"], 120)
        self.assertEqual(diagnostic["frame_size"], "640x360")
        self.assertEqual(diagnostic["hidden_by_recognized_only"], 1)

    def test_unread_vehicles_are_excluded_from_results_and_csv(self):
        for plate, status in (("", "low"), ("   ", "low"), (None, "low"),
                              ("29A12345", "confirmed")):
            self.store.append({"Camera": "test", "Biển số": plate,
                               "confidence_status": status})
        rows = self.client.get("/api/events").json
        self.assertEqual([row["Biển số"] for row in rows], ["29A12345"])
        self.assertEqual(self.client.get("/api/status").json["hidden_by_plates_only"], 3)
        with self.client.get("/api/events.csv") as response:
            self.assertEqual(len(response.data.decode("utf-8-sig").splitlines()), 2)

    def test_results_always_include_only_plate_rows(self):
        self.store.append({"Camera": "test", "Track ID": 1, "Biển số": "29A12345",
                           "confidence_status": "confirmed"})
        self.store.append({"Camera": "test", "Track ID": 2, "Biển số": "30A99999",
                           "confidence_status": "low"})
        self.store.append({"Camera": "test", "Track ID": 3, "Biển số": "",
                           "confidence_status": "low"})
        self.assertEqual([row["Biển số"] for row in self.client.get("/api/events").json],
                         ["30A99999", "29A12345"])
        self.assertEqual(self.client.get("/api/status").json["hidden_by_plates_only"], 1)
        with patch("vehicle_ai.web.Thread") as thread:
            self.client.post("/api/analyze", data={"url": "rtsp://test"})
        run = thread.call_args.kwargs["target"]
        config = PipelineConfig(vehicle_weights="v", roi=[[0, 0], [1, 0], [1, 1], [0, 1]])

        def analyze(*args, **kwargs):
            kwargs["on_event"]({"Biển số": ""})

        with (
            patch("vehicle_ai.web_models.prepare_models", return_value=(config, None, None, None)),
            patch("vehicle_ai.capture.stream_source", return_value=("test", "test")),
            patch("vehicle_ai.pipeline.analyze_video", side_effect=analyze),
        ):
            run(*thread.call_args.kwargs["args"])
        status = self.client.get("/api/status").json
        self.assertTrue(status["plates_only"])
        rows = self.client.get("/api/events").json
        self.assertEqual([row["Biển số"] for row in rows], ["30A99999", "29A12345"])
        self.assertEqual(self.client.get("/api/status").json["hidden_by_plates_only"], 1)
        with self.client.get("/api/events.csv") as response:
            text = response.data.decode("utf-8-sig")
        self.assertIn("29A12345", text)
        self.assertIn("30A99999", text)
        csv_rows = list(csv.DictReader(StringIO(text)))
        self.assertEqual(len(csv_rows), 2)
        self.assertTrue(all(row["Biển số"] for row in csv_rows))

    def test_live_preview_is_multipart_jpeg(self):
        with patch("vehicle_ai.web.Thread") as thread:
            self.client.post("/api/analyze", data={"url": "rtsp://test"})
        run = thread.call_args.kwargs["target"]

        def analyze(*args, **kwargs):
            import numpy as np
            self.assertTrue(kwargs["realtime"])
            self.assertFalse(kwargs["recognized_only"])
            kwargs["on_frame"](1, 10, np.zeros((24, 32, 3), dtype=np.uint8))
            kwargs["on_event"]({"Biển số": ""})

        with (
            patch("vehicle_ai.web_models.prepare_models", return_value=(PipelineConfig(vehicle_weights="v"), None, None, None)),
            patch("vehicle_ai.capture.stream_source", return_value=("test", "test")),
            patch("vehicle_ai.pipeline.analyze_video", side_effect=analyze),
        ):
            run(*thread.call_args.kwargs["args"])
        self.assertEqual(self.store.recent(), [])
        with self.client.get("/api/stream", buffered=False) as response:
            self.assertIn("multipart/x-mixed-replace", response.content_type)
            chunk = next(response.response)
            self.assertIn(b"Content-Type: image/jpeg", chunk)
            self.assertIn(b"\xff\xd8", chunk)

    def test_evidence_download_and_path_boundary(self):
        event = self.store.append({"Track ID": 1, "Biển số": "29A12345"})
        photo = self.root / "plate.jpg"
        photo.write_bytes(b"jpeg")
        self.store.update_images(event["event_id"], {"plate": [{"path": str(photo)}]})
        rows = self.client.get("/api/events").json
        self.assertNotIn("images", rows[0])
        url = rows[0]["photos"]["plate"]
        with patch("vehicle_ai.web.image_root", return_value=self.root):
            response = self.client.get(url + "&download=1")
            self.assertEqual(response.data, b"jpeg")
            self.assertIn("attachment", response.headers["Content-Disposition"])
            response.close()
        with patch("vehicle_ai.web.image_root", return_value=self.root / "other"):
            self.assertEqual(self.client.get(url).status_code, 404)


if __name__ == "__main__":
    unittest.main()
