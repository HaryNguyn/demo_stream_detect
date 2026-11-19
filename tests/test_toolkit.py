import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock

import cv2
import numpy as np
import torch
from ultralytics.engine.results import Results

from vehicle_ai.config import PipelineConfig
from vehicle_ai.datasets import build_yolo, split_manifest, validate_manifest, write_manifest
from vehicle_ai.evaluation import evaluate_events
from vehicle_ai.session import RecognitionSession
from vehicle_ai.storage import ExperimentWriter
from vehicle_ai.crnn import decode


class SessionTests(unittest.TestCase):
    def make_session(self):
        frame = np.full((80, 120, 3), 127, np.uint8)
        found = Results(frame, path="", names={0: "car"},
                        boxes=torch.tensor([[5, 5, 100, 70, 1, 0.9, 0]]))
        empty = Results(frame, path="", names={0: "car"}, boxes=torch.empty((0, 7)))
        detector = MagicMock()
        detector.track.side_effect = [[found], [empty]]
        config = PipelineConfig(vehicle_weights="unused", vehicle_classes={0: "car"})
        return RecognitionSession(detector, config, "cam", source_id="video"), frame

    def test_timeout_outputs_event_once_and_saves_evidence(self):
        session, frame = self.make_session()
        self.assertEqual(session.feed(frame, 0), [])
        events = session.feed(frame, 3)
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event["event"]["status"], "no_plate")
        self.assertEqual(event["event"]["vehicle_type"], "car")
        self.assertEqual(event["event"]["finalization_reason"], "track_timeout")
        self.assertEqual(event["event"]["ended_at_seconds"], 0)
        self.assertEqual(session.finalize(), [])
        with self.assertRaises(RuntimeError):
            session.feed(frame, 4)
        with tempfile.TemporaryDirectory() as tmp:
            writer = ExperimentWriter(Path(tmp) / "run")
            writer.append(event)
            writer.close()
            saved = json.loads((Path(tmp) / "run" / "events.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(len(saved["evidence"]), 1)
            self.assertIsNotNone(cv2.imread(str(Path(tmp) / "run" / saved["evidence"][0]["path"])))
            self.assertNotIn("base64", json.dumps(saved))

    def test_finalize_flushes_and_is_idempotent(self):
        session, frame = self.make_session()
        session.feed(frame, 1)
        with self.assertRaises(ValueError):
            session.feed(frame, 0.5)
        with self.assertRaises(ValueError):
            session.feed(frame, float("nan"))
        self.assertEqual(len(session.finalize()), 1)
        self.assertEqual(session.finalize(), [])


class EvaluationTests(unittest.TestCase):
    def event(self, identity, start=0, end=10, plate="30A12345", source="video"):
        return {"event_id": identity, "source_id": source, "camera_id": "cam",
                "started_at_seconds": start, "ended_at_seconds": end,
                "plate": plate, "vehicle_type": "car", "brand": "Toyota"}

    def test_duplicate_does_not_inflate_recall_and_wrong_plate_still_matches(self):
        truth = [self.event("a"), self.event("b", 20, 30)]
        predictions = [self.event("p1", plate="WRONG"), self.event("p2", 1, 9)]
        report = evaluate_events(truth, predictions)
        self.assertEqual(report["matched_events"], 1)
        self.assertEqual(report["possible_duplicate_events"], 1)
        self.assertEqual(report["missed_events"], 1)
        self.assertEqual(report["plate_exact_matches"], 0)
        self.assertEqual(report["event_precision"], 0.5)

    def test_no_predictions_and_cross_source(self):
        truth = [self.event("a")]
        self.assertEqual(evaluate_events(truth, [])["plate_end_to_end_recall"], 0)
        report = evaluate_events(truth, [self.event("p", source="other")])
        self.assertEqual(report["matched_events"], 0)
        with self.assertRaises(ValueError):
            evaluate_events([], [])


class DatasetTests(unittest.TestCase):
    def create_data(self, root):
        rows = []
        for i, split in enumerate(("train", "val", "test")):
            image = root / f"{i}.png"
            cv2.imwrite(str(image), np.full((32, 64, 3), 20 + i * 60, np.uint8))
            label = root / f"{i}.txt"
            label.write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
            rows.append({"image": str(image), "label": str(label), "text": "30A11111",
                         "group_id": str(i), "split": split})
        return rows

    def test_validate_and_build_detector_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = self.create_data(root)
            manifest = root / "manifest.csv"
            write_manifest(manifest, rows)
            self.assertEqual(validate_manifest(manifest, "ocr")["images"], 3)
            result = build_yolo(manifest, root / "yolo", ["car"])
            self.assertEqual(result["names"], ["car"])
            self.assertEqual(len(list((root / "yolo" / "labels").rglob("*.txt"))), 3)
            self.assertEqual(split_manifest(manifest, root / "split.csv"), {"train": 1, "val": 1, "test": 1})

    def test_reject_group_and_pixel_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = self.create_data(root)
            rows[1]["group_id"] = rows[0]["group_id"]
            write_manifest(root / "groups.csv", rows)
            with self.assertRaisesRegex(ValueError, "group_id"):
                validate_manifest(root / "groups.csv", "ocr")
            rows[1]["group_id"] = "val"
            cv2.imwrite(rows[1]["image"], cv2.imread(rows[0]["image"]))
            write_manifest(root / "pixels.csv", rows)
            with self.assertRaisesRegex(ValueError, "trùng nội dung"):
                validate_manifest(root / "pixels.csv", "ocr")

    def test_reject_invalid_box(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = self.create_data(root)
            Path(rows[0]["label"]).write_text("0 .1 .1 .9 .9", encoding="utf-8")
            write_manifest(root / "manifest.csv", rows)
            with self.assertRaisesRegex(ValueError, "Bbox"):
                validate_manifest(root / "manifest.csv", "detect", ["car"])


class OCRTests(unittest.TestCase):
    def test_ctc_repeats_require_blank_separator(self):
        # 0=blank, 1=A, 2=B. AA(blank)ABB -> AAB.
        logits = torch.full((7, 1, 3), -10.0)
        for step, token in enumerate([1, 1, 0, 1, 2, 2, 0]):
            logits[step, 0, token] = 10
        self.assertEqual(decode(logits, "AB")[0]["rec_text"], "AAB")

    def test_crnn_one_epoch_checkpoint_roundtrip(self):
        from vehicle_ai.training import train_crnn
        from vehicle_ai.crnn import CRNNRecognizer

        original_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                rows = DatasetTests().create_data(root)
                write_manifest(root / "manifest.csv", rows)
                result = train_crnn(root / "manifest.csv", root / "train", epochs=1, batch=1)
                recognizer = CRNNRecognizer(result["best_checkpoint"])
                prediction = recognizer.predict(cv2.imread(rows[0]["image"]))
                self.assertIn("rec_text", prediction[0])
                self.assertEqual(len((root / "train" / "history.jsonl").read_text().splitlines()), 1)
        finally:
            torch.set_num_threads(original_threads)


if __name__ == "__main__":
    unittest.main()
