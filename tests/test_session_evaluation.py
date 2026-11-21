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
from vehicle_ai.evaluation import evaluate_events
from vehicle_ai.session import RecognitionSession
from vehicle_ai.storage import ExperimentWriter


class SessionTests(unittest.TestCase):
    def make_session(self):
        frame = np.full((80, 120, 3), 127, np.uint8)
        found = Results(frame, path="", names={0: "car"},
                        boxes=torch.tensor([[5, 5, 100, 70, 1, 0.9, 0]]))
        empty = Results(frame, path="", names={0: "car"}, boxes=torch.empty((0, 7)))
        detector = MagicMock()
        detector.track.side_effect = [[found], [empty]]
        # Small synthetic crop: exercise evidence storage without the production size gate.
        config = PipelineConfig(vehicle_weights="unused", vehicle_classes={0: "car"}, min_vehicle_width=0)
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


if __name__ == "__main__":
    unittest.main()
