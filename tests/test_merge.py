import io
import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from vehicle_ai import capture as sources
from vehicle_ai.aggregation import MAX_IMAGE_SAMPLES, Vehicle
from vehicle_ai.persistence import create_event_store, MariaDBEventStore


class StorageTests(unittest.TestCase):
    def test_existing_sqlite_history_and_new_images(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.sqlite3"
            with sqlite3.connect(path) as db:
                db.execute("CREATE TABLE events (event_id TEXT PRIMARY KEY, recorded_at TEXT, payload TEXT)")
                db.execute("INSERT INTO events VALUES (?, ?, ?)",
                           ("old", "2025", json.dumps({"event_id": "old", "Camera": "cũ"})))
            db.close()
            store = create_event_store(path)
            event = store.append({
                "Camera": "Mới", "Track ID": 12, "Biển số": "30A12345",
                "Danh sách ảnh xe": [{"data": "BASE64_SENTINEL"}],
            })
            saved = {"detected": [{"path": "vehicle.jpg"}], "plate": []}
            store.update_images(event["event_id"], saved)
            reopened = create_event_store(path)
            rows = reopened.recent()
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["images"], saved)
            self.assertEqual(rows[0]["Xe ID"], "30A12345")
            self.assertEqual(rows[1]["event_id"], "old")
            self.assertNotIn("BASE64_SENTINEL", json.dumps(rows))
            self.assertEqual(len(reopened.recent(1)), 1)
            self.assertIn("Danh sách ảnh xe", event)

    def test_invalid_backend(self):
        with patch.dict(os.environ, {"EVENT_STORE": "invalid"}):
            with self.assertRaises(ValueError):
                create_event_store()

    def test_mariadb_creates_schema_and_missing_identity_column(self):
        connection = MagicMock()
        cursor = connection.__enter__.return_value.cursor.return_value.__enter__.return_value
        cursor.fetchone.return_value = None
        with patch.dict(os.environ, {"MARIADB_USER": "test", "MARIADB_PASSWORD": "test"}):
            with patch.object(MariaDBEventStore, "connect", return_value=connection):
                MariaDBEventStore()
        statements = [call.args[0] for call in cursor.execute.call_args_list]
        self.assertIn("xe_id VARCHAR(255)", statements[0])
        self.assertTrue(any("ALTER TABLE" in sql for sql in statements))


class SamplingTests(unittest.TestCase):
    def test_long_track_is_bounded_and_exports_three_distinct_frames(self):
        vehicle = Vehicle(time=0)
        for index in range(1000):
            vehicle.add_sample("detected", {
                "_frame_idx": index * 5, "_score": index / 1000,
                "frame_idx": index * 5, "data": "jpeg",
            })
        self.assertEqual(len(vehicle.detected_samples), MAX_IMAGE_SAMPLES)
        row = vehicle.row("camera", 1)
        samples = row["Danh sách ảnh xe"]
        self.assertEqual(len(samples), 3)
        self.assertEqual(samples[-1]["frame_idx"], 4995)
        self.assertFalse(any(key.startswith("_") for sample in samples for key in sample))


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


if __name__ == "__main__":
    unittest.main()
