"""Application boundaries: settings, evidence persistence and web/review entry points."""
import base64
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from vehicle_ai.evidence import save_event_images
from event_store_stub import EventStoreStub
from vehicle_ai.settings import get_setting


class SettingsTests(unittest.TestCase):
    def test_environment_overrides_dotenv_without_mutating_environment(self):
        key = "VEHICLE_AI_TEST_SETTING"
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp) / ".env").write_text(f"{key}=from-file\n", encoding="utf-8")
            with patch.dict(os.environ, {"VEHICLE_AI_HOME": temp}):
                os.environ.pop(key, None)
                self.assertEqual(get_setting(key), "from-file")
                self.assertNotIn(key, os.environ)
                with patch.dict(os.environ, {key: "from-env"}):
                    self.assertEqual(get_setting(key), "from-env")


class EvidenceTests(unittest.TestCase):
    def test_evidence_roundtrip_without_mutating_input(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = EventStoreStub()
            image = np.zeros((12, 24, 3), np.uint8)
            ok, encoded = cv2.imencode(".jpg", image)
            self.assertTrue(ok)
            sample = {"encoding": "base64", "data": base64.b64encode(encoded).decode(),
                      "width": 24, "height": 12}
            row = {"Camera": "CON", "Track ID": 1, "Danh sách ảnh xe": [sample]}
            event = store.append(row)
            folder, metadata_path, saved = save_event_images(event, root / "images")
            store.update_images(event["event_id"], saved)
            self.assertIn("_CON", folder.parts)
            self.assertIn("Danh sách ảnh xe", event)
            self.assertEqual(store.recent()[0]["images"], saved)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertNotIn("Danh sách ảnh xe", metadata)
            self.assertEqual(cv2.imread(saved["detected"][0]["path"]).shape, image.shape)

    def test_locked_obsolete_image_does_not_abort_save(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "images"
            ok, encoded = cv2.imencode(".jpg", np.zeros((12, 24, 3), np.uint8))
            self.assertTrue(ok)
            sample = {"encoding": "base64", "data": base64.b64encode(encoded).decode(),
                      "width": 24, "height": 12}
            row = {"Camera": "cam", "Track ID": 1, "event_id": "evt-1",
                   "Danh sách ảnh xe": [sample]}
            _, _, saved = save_event_images(row, root)
            old_image = saved["detected"][0]
            self.assertTrue(Path(old_image["path"]).is_file())
            with patch.object(Path, "unlink", side_effect=PermissionError("locked")):
                _, metadata_path, saved_again = save_event_images(
                    {"Camera": "cam", "Track ID": 1, "event_id": "evt-1",
                     "Danh sách ảnh xe": []},
                    root, previous_images={"detected": [old_image]},
                )
            self.assertEqual(saved_again["detected"], [])
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["pending_image_cleanup"], [old_image["path"]])
            self.assertTrue(Path(old_image["path"]).is_file())


class UITests(unittest.TestCase):
    def test_detection_entrypoint_uses_web_server(self):
        import app

        store = EventStoreStub()
        web = app.create_app(store)
        with web.test_client() as client:
            with client.get("/") as response:
                self.assertEqual(response.status_code, 200)
                self.assertIn(b'/static/app.js', response.data)
            self.assertEqual(client.get("/api/status").json["status"], "idle")
        with patch("app.create_app", return_value=web), patch.object(web, "run") as run:
            with patch("sys.argv", ["app.py", "--port", "8002"]):
                app.main()
            run.assert_called_once_with(host="127.0.0.1", port=8002, threaded=True)

    def test_image_endpoint_serves_evidence_without_file_handle(self):
        import app

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ok, encoded = cv2.imencode(".jpg", np.zeros((12, 24, 3), np.uint8))
            self.assertTrue(ok)
            image_path = root / "detected_test.jpg"
            image_path.write_bytes(encoded.tobytes())
            store = EventStoreStub()
            store.rows.append({"event_id": "evt-img", "images": {"detected": [
                {"path": str(image_path), "mime_type": "image/jpeg", "quality_score": 1.0}
            ]}})
            web = app.create_app(store)
            with patch("vehicle_ai.web.image_root", return_value=root):
                with web.test_client() as client:
                    response = client.get("/api/events/evt-img/images/detected")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.data, encoded.tobytes())
            image_path.unlink()


if __name__ == "__main__":
    unittest.main()
