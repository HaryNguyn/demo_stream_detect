import csv
from pathlib import Path
import tempfile
import unittest

from vehicle_ai.annotation import approve_record, atomic_json, validate_boxes


class AnnotationTests(unittest.TestCase):
    def test_only_approved_enter_manifest_and_skip_removes_them(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            atomic_json(root / "review.json", {
                "generation_status": "completed",
                "records": [
                    {"id": "a", "image": "a.jpg", "group_id": "video1", "status": "pending"},
                    {"id": "b", "image": "b.jpg", "group_id": "video2", "status": "pending"},
                ],
            })
            boxes = [{"class_id": 0, "x_center": .5, "y_center": .5, "width": .5, "height": .5}]
            approve_record(root, "a", boxes)
            with (root / "reviewed_manifest.csv").open(encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["group_id"], "video1")
            self.assertTrue(Path(rows[0]["label"]).is_file())
            with self.assertRaises(ValueError):
                approve_record(root, "b", [])
            approve_record(root, "b", [], confirm_empty=True)
            approve_record(root, "a", [], action="skip")
            with (root / "reviewed_manifest.csv").open(encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([r["group_id"] for r in rows], ["video2"])
            self.assertEqual(Path(rows[0]["label"]).read_text(), "")

    def test_invalid_boxes_are_not_accepted(self):
        with self.assertRaises(ValueError):
            validate_boxes([{"class_id": 4, "x_center": .5, "y_center": .5, "width": .5, "height": .5}])
        with self.assertRaises(ValueError):
            validate_boxes([{"class_id": 0, "x_center": .1, "y_center": .5, "width": .5, "height": .5}])


if __name__ == "__main__":
    unittest.main()
