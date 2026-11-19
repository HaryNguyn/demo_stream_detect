"""Application boundaries: settings, evidence persistence and Streamlit entry points."""
import base64
import csv
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from streamlit.testing.v1 import AppTest

from vehicle_ai.evidence import save_event_images
from vehicle_ai.persistence import SQLiteEventStore
from vehicle_ai.settings import get_setting

ROOT = Path(__file__).resolve().parents[1]


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
    def test_sqlite_and_evidence_roundtrip_without_mutating_input(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = SQLiteEventStore(root / "events.sqlite3")
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


class UITests(unittest.TestCase):
    def test_detection_entrypoint_reruns_with_temporary_database(self):
        with tempfile.TemporaryDirectory() as temp:
            store = SQLiteEventStore(Path(temp) / "events.sqlite3")
            with patch("vehicle_ai.ui.detection.create_event_store", return_value=store):
                app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=60)
                self.assertFalse(app.exception)
                app.checkbox[0].check().run()
                self.assertFalse(app.exception)
                self.assertTrue(app.button[0].disabled)

    def test_review_entrypoint_approves_only_temporary_record(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = root / "data/review/vehicle-v001"
            directory.mkdir(parents=True)
            image_path = root / "sample.jpg"
            cv2.imwrite(str(image_path), np.zeros((64, 128, 3), np.uint8))
            boxes = [{"class_id": 0, "x_center": .5, "y_center": .5, "width": .5, "height": .5}]
            project = {
                "generation_status": "completed",
                "records": [{"id": "sample", "image": str(image_path), "group_id": "video",
                             "status": "pending", "suggested_boxes": boxes}],
            }
            (directory / "review.json").write_text(json.dumps(project), encoding="utf-8")
            with patch("vehicle_ai.ui.review.project_dir", return_value=root):
                app = AppTest.from_file(str(ROOT / "review_app.py")).run(timeout=60)
                self.assertFalse(app.exception)
                app.button[0].click().run()
                self.assertFalse(app.exception)
            with (directory / "reviewed_manifest.csv").open(encoding="utf-8") as stream:
                approved = list(csv.DictReader(stream))
            self.assertEqual(len(approved), 1)
            self.assertEqual(approved[0]["group_id"], "video")


if __name__ == "__main__":
    unittest.main()
