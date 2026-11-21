import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from event_store_stub import EventStoreStub
from test_evidence_quality import event
from vehicle_ai.config import PipelineConfig
from vehicle_ai.consensus import classification_decisions
from vehicle_ai.domain import event_from_row
from vehicle_ai.evaluation import compare_reports, evaluate_events, read_events
from vehicle_ai.result_filter import EvidenceRecorder, unique_results
from vehicle_ai.storage import ExperimentWriter, MergedExperimentStore


def sample(identity, start, plate, status=None, end=None):
    row = {"event_id": identity, "source_id": "source", "camera_id": "cam",
           "started_at_seconds": start, "ended_at_seconds": start + 10 if end is None else end, "plate": plate}
    if status:
        row["confidence_status"] = status
    return row


class ConfidenceEvaluationTests(unittest.TestCase):
    def test_groups_use_global_matching_without_matching_duplicate_twice(self):
        truth = [sample("t1", 0, "29A12345"), sample("t2", 20, "30A12345")]
        predictions = [sample("p1", 0, "29A12345", "confirmed"),
                       sample("p2", 20, "30A12346", "low"),
                       sample("duplicate", 1, "29A12345", "low", end=9)]
        report = evaluate_events(truth, predictions)
        confirmed, low = report["confidence_groups"]["confirmed"], report["confidence_groups"]["low"]
        self.assertEqual(confirmed["plate_exact_match_on_matched"], 1)
        self.assertEqual(low["plate_exact_match_on_matched"], 0)
        self.assertEqual(low["possible_duplicate_events"], 1)
        self.assertEqual(low["matched_events"], 1)
        self.assertAlmostEqual(low["plate_cer_on_matched"], 1 / 8)
        self.assertAlmostEqual(report["confirmation_rate"], 1 / 3)

    def test_unlabelled_legacy_events_are_unknown_and_no_baseline_rate_is_invented(self):
        truth = [sample("t", 0, "29A12345")]
        before = evaluate_events(truth, [sample("p", 0, "29A12346")])
        after = evaluate_events(truth, [sample("p", 0, "29A12345", "confirmed")])
        delta = compare_reports(before, after)
        self.assertIsNone(before["confirmation_rate"])
        self.assertIsNone(delta["confirmation_rate"])
        self.assertEqual(delta["plate_exact_matches"], 1)
        self.assertLess(delta["plate_cer_on_matched"], 0)

    def test_classification_has_its_own_minimum_reads_and_share(self):
        config = PipelineConfig(vehicle_weights="unused")
        summaries = {"brand": {"brandA": {"score": 1.9, "count": 2, "confidence_sum": 1.9}},
                     "vehicle_type": {"car": {"score": 3.6, "count": 4, "confidence_sum": 3.6}}}
        decisions = classification_decisions(summaries, config)
        self.assertFalse(decisions["brand"]["confirmed"])
        self.assertTrue(decisions["vehicle_type"]["confirmed"])

    def test_track_switch_merges_votes_once_and_promotes_plate(self):
        with TemporaryDirectory() as directory:
            store = EventStoreStub()
            recorder = EvidenceRecorder(store, directory, keep_low_confidence=True)
            def row(track):
                item = event(.9, track=track)
                item["confidence_status"] = "low"
                item["source_track_ids"] = [track]
                item["vote_summaries"] = {
                    "plate": {"29A12345": {"score": 1.9, "count": 2, "confidence_sum": 1.9}},
                    "brand": {"Toyota": {"score": 1.5, "count": 2, "confidence_sum": 1.8}},
                    "vehicle_type": {"car": {"score": 1.8, "count": 2, "confidence_sum": 1.8}},
                }
                return item
            recorder.record(row(1))
            self.assertEqual(unique_results(store.recent())[0]["confidence_status"], "low")
            recorder.record(row(2))
            recorder.record(row(2))
            results = unique_results(store.recent())
            self.assertEqual(len(results), 1)
            merged = results[0]
            self.assertEqual(merged["confidence_status"], "confirmed")
            self.assertEqual(merged["vote_summaries"]["plate"]["29A12345"]["count"], 4)
            self.assertTrue(merged["classification_status"]["brand"]["confirmed"])
            self.assertEqual(merged["source_track_ids"], [1, 2])

    def test_experiment_output_retains_confidence_group(self):
        row = {"Camera": "cam", "Track ID": 1, "Xuất hiện (giây)": 0, "Kết thúc (giây)": 1,
               "Loại xe": "car", "Hãng": "", "Điểm hãng": 0, "Biển số": "29A12345", "Điểm biển": .9,
               "status": "recognized", "confidence_status": "low",
               "plate_decision": {"unmet": ["score"]}}
        row["event"] = event_from_row(row, source_id="s", model_version="m", reason="end")
        with TemporaryDirectory() as directory:
            writer = ExperimentWriter(Path(directory) / "run")
            writer.append(row)
            writer.close()
            saved = json.loads((writer.directory / "events.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(saved["confidence_status"], "low")
            self.assertEqual(saved["plate_decision"]["unmet"], ["score"])

    def test_brand_comparison_ignores_case_and_separates_unknown_taxonomy(self):
        truth = [dict(sample("t1", 0, "29A12345"), brand="Toyota", vehicle_type="Ô tô con"),
                 dict(sample("t2", 20, "30A12345"), brand="VinFast", vehicle_type="Ô tô con")]
        predictions = [
            dict(sample("p1", 0, "29A12345", "confirmed"), brand="toyota",
                 vehicle_type="ô tô con", classification_status={"brand": {"confirmed": True}}),
            dict(sample("p2", 20, "30A12345", "confirmed"), brand="toyota",
                 vehicle_type="Ô tô con", classification_status={"brand": {"confirmed": True}}),
        ]
        report = evaluate_events(truth, predictions, known_brands=["toyota", "bmw"])
        self.assertEqual(report["field_counts"]["brand"]["correct_over_total"], "1/2")
        self.assertEqual(report["brand_groups"]["known"]["correct_over_total"], "1/1")
        self.assertEqual(report["brand_groups"]["unknown"]["correct_over_total"], "0/1")
        self.assertEqual(report["brand_groups"]["unknown"]["confident_wrong"], 1)
        self.assertAlmostEqual(report["brand_groups"]["unknown"]["accuracy_on_matched"], 0.0)

        unspecified = evaluate_events(truth, predictions)
        self.assertEqual(unspecified["brand_groups"]["taxonomy_unspecified"]["matched_events"], 2)
        self.assertEqual(unspecified["brand_groups"]["taxonomy_unspecified"]["correct_over_total"], "1/2")

    def test_read_events_accepts_null_and_optional_metadata(self):
        rows = [
            dict(sample("g1", 0, None), vehicle_type=None, brand=None, tags=["night", "hard"], notes=""),
            dict(sample("g2", 10, "30A12345"), vehicle_type="Ô tô con", brand="toyota"),
        ]
        rows[1].pop("plate")
        with TemporaryDirectory() as directory:
            path = Path(directory) / "truth.jsonl"
            path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
            loaded = read_events(path)
        self.assertIsNone(loaded[0]["plate"])
        self.assertNotIn("plate", loaded[1])
        self.assertIsNone(loaded[0]["brand"])
        self.assertEqual(loaded[0]["tags"], ["night", "hard"])

    def test_read_events_rejects_malformed_tags_and_notes(self):
        row = dict(sample("g1", 0, "30A12345"), tags="night")
        with TemporaryDirectory() as directory:
            path = Path(directory) / "truth.jsonl"
            path.write_text(json.dumps(row), encoding="utf-8")
            with self.assertRaises(ValueError):
                read_events(path)
            row["tags"] = ["night", ""]
            path.write_text(json.dumps(row), encoding="utf-8")
            with self.assertRaises(ValueError):
                read_events(path)
            row["tags"] = ["night"]
            row["notes"] = 5
            path.write_text(json.dumps(row), encoding="utf-8")
            with self.assertRaises(ValueError):
                read_events(path)

    def test_tag_groups_split_recall_and_exclude_unread_plates(self):
        truth = [
            dict(sample("t1", 0, "29A12345"), tags=["day", "car"]),
            dict(sample("t2", 20, "30A12345"), tags=["night", "car"]),
            dict(sample("t3", 40, None), tags=["night", "hard"]),
        ]
        predictions = [sample("p1", 0, "29A12345", "confirmed"),
                       sample("p2", 20, "30A12346", "low")]
        report = evaluate_events(truth, predictions)
        night = report["tag_groups"]["night"]
        self.assertEqual(night["ground_truth_events"], 2)
        self.assertEqual(night["matched_events"], 1)
        self.assertEqual(night["missed_events"], 1)
        self.assertAlmostEqual(night["event_recall"], 0.5)
        self.assertEqual(night["plate_correct_over_total"], "0/1")
        hard = report["tag_groups"]["hard"]
        self.assertIsNone(hard["plate_correct_over_total"])
        self.assertAlmostEqual(hard["event_recall"], 0.0)
        day = report["tag_groups"]["day"]
        self.assertEqual(day["plate_correct_over_total"], "1/1")
        self.assertAlmostEqual(day["plate_end_to_end_recall"], 1.0)

    def test_compare_reports_includes_tag_delta(self):
        truth = [dict(sample("t1", 0, "29A12345"), tags=["day"]),
                 dict(sample("t2", 20, "30A12345"), tags=["night"])]
        before = evaluate_events(truth, [sample("p1", 0, "29A12345")])
        after = evaluate_events(truth, [sample("p1", 0, "29A12345"), sample("p2", 20, "30A12345")])
        delta = compare_reports(before, after)
        self.assertEqual(delta["tag_groups"]["day"]["matched_events"], 0)
        self.assertEqual(delta["tag_groups"]["night"]["matched_events"], 1)
        self.assertAlmostEqual(delta["tag_groups"]["night"]["event_recall"], 1.0)

    def test_merged_export_marks_stage_and_keeps_raw_event_ids(self):
        def full_row(track):
            item = event(.9, track=track)
            item.update({
                "Xuất hiện (giây)": float(track), "Kết thúc (giây)": float(track) + 2,
                "Loại xe": "Ô tô con", "Hãng": "", "Điểm hãng": 0, "Điểm biển": .95,
                "status": "recognized", "confidence_status": "low",
                "source_track_ids": [track], "raw_event_ids": [f"raw-{track}"],
                "vote_summaries": {
                    "plate": {"29A12345": {"score": 1.9, "count": 2, "confidence_sum": 1.9}},
                    "brand": {}, "vehicle_type": {},
                },
            })
            return item

        with TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            root.mkdir()
            store = MergedExperimentStore(root, "source", "model")
            recorder = EvidenceRecorder(store, root / "evidence", keep_low_confidence=True)
            recorder.record(full_row(1))
            recorder.record(full_row(2))
            store.close()
            rows = [json.loads(line) for line in
                    (root / "events.merged.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 1)
            merged = rows[0]
            self.assertEqual(merged["event_stage"], "merged")
            self.assertEqual(merged["raw_event_ids"], ["raw-1", "raw-2"])
            self.assertEqual(merged["source_track_ids"], [1, 2])
            self.assertEqual(merged["confidence_status"], "confirmed")
            self.assertEqual(merged["vote_summaries"]["plate"]["29A12345"]["count"], 4)
            self.assertTrue(merged["evidence"])
            self.assertTrue(Path(root / merged["evidence"][0]["path"]).is_file())

    def test_merged_export_keeps_rejection_reasons_from_both_tracks(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            root.mkdir()
            store = MergedExperimentStore(root, "source", "model")
            recorder = EvidenceRecorder(store, root / "evidence", keep_low_confidence=True)
            low = event(.2, track=1)
            low.update({
                "Xuất hiện (giây)": 0.0, "Kết thúc (giây)": 2.0, "Loại xe": "Ô tô con",
                "Hãng": "", "Điểm hãng": 0, "Điểm biển": .2, "status": "recognized",
                "confidence_status": "low",
                "source_track_ids": [1],
                "rejection_reasons": {"plate_not_confirmed": 2},
                "rejection_detail": {"plate_not_confirmed": "score"},
            })
            high = event(.9, track=2)
            high.update({
                "Xuất hiện (giây)": 1.0, "Kết thúc (giây)": 3.0, "Loại xe": "Ô tô con",
                "Hãng": "", "Điểm hãng": 0, "Điểm biển": .95, "status": "recognized",
                "confidence_status": "confirmed",
                "source_track_ids": [2],
                "vote_summaries": {
                    "plate": {"29A12345": {"score": 3.8, "count": 4, "confidence_sum": 3.8}},
                    "brand": {}, "vehicle_type": {},
                },
                "rejection_reasons": {"plate_too_small": 1},
                "rejection_detail": {"plate_too_small": "28px < 32px"},
            })
            recorder.record(low)
            recorder.record(high)
            store.close()
            merged = json.loads((root / "events.merged.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(merged["rejection_reasons"]["plate_not_confirmed"], 2)
            self.assertEqual(merged["rejection_reasons"]["plate_too_small"], 1)
            self.assertEqual(merged["rejection_reasons"]["merged_duplicate"], 1)
            self.assertEqual(merged["rejection_detail"]["plate_too_small"], "28px < 32px")
            self.assertEqual(merged["rejection_detail"]["merged_duplicate"], "gộp vào " + merged["event_id"])
