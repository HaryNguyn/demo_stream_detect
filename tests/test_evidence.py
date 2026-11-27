"""Event persistence, evidence quality, and confidence across tracks."""

import base64
import json
import os
import tempfile
import unittest
from base64 import b64encode
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from event_store_stub import EventStoreStub

from vehicle_ai.aggregation import MAX_IMAGE_SAMPLES, Vehicle
from vehicle_ai.config import PipelineConfig
from vehicle_ai.consensus import classification_decisions
from vehicle_ai.domain import event_from_row
from vehicle_ai.evaluation import compare_reports, evaluate_events, read_events
from vehicle_ai.evidence import save_event_images
from vehicle_ai.persistence import MariaDBEventStore, create_event_store
from vehicle_ai.quality import plate_quality
from vehicle_ai.result_filter import EvidenceRecorder, unique_results
from vehicle_ai.storage import ExperimentWriter, MergedExperimentStore


class StorageTests(unittest.TestCase):
    def test_mariadb_writes_metadata_and_image_paths(self):
        connection = MagicMock()
        cursor = connection.__enter__.return_value.cursor.return_value.__enter__.return_value
        with patch.dict(os.environ, {"MARIADB_USER": "test", "MARIADB_PASSWORD": "test"}):
            with patch.object(MariaDBEventStore, "connect", return_value=connection):
                store = MariaDBEventStore()
                cursor.execute.reset_mock()
                row = {"Camera": "Mới", "Track ID": 12, "Biển số": "30A12345",
                       "Danh sách ảnh xe": [{"data": "BASE64_SENTINEL"}]}
                event = store.append(row)
                sql, values = cursor.execute.call_args.args
                self.assertIn("INSERT INTO events", sql)
                self.assertEqual(values[3], "30A12345")
                self.assertNotIn("BASE64_SENTINEL", str(values))
                self.assertIn("Danh sách ảnh xe", event)
                self.assertNotIn("event_id", row)
                saved = {"detected": [{"path": "vehicle.jpg"}], "plate": []}
                store.update_images(event["event_id"], saved)
                sql, values = cursor.execute.call_args.args
                self.assertIn("UPDATE events", sql)
                self.assertEqual(json.loads(values[0]), saved)
                self.assertEqual(values[1], event["event_id"])

    def test_factory_always_uses_mariadb(self):
        with patch("vehicle_ai.persistence.MariaDBEventStore") as store:
            self.assertIs(create_event_store(), store.return_value)
            store.assert_called_once_with()

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


def photo(quality=.8, detection=.9, ocr=.95, plate="29A12345"):
    return {
        "encoding": "base64", "mime_type": "image/jpeg", "data": b64encode(b"image").decode(),
        "quality_score": quality, "detection_score": detection, "ocr_score": ocr,
        "recognized_plate": plate, "_score": quality, "_frame_idx": 1,
    }


def event(quality=.8, track=1, plate="29A12345", camera="camera"):
    return {
        "Camera": camera, "Track ID": track, "Biển số": plate,
        "Danh sách ảnh biển": [photo(quality, plate=plate)],
        "Danh sách ảnh xe": [photo(.7)],
    }


class EvidenceQualityTests(unittest.TestCase):
    def test_only_best_qualified_image_matching_plate_is_selected(self):
        vehicle = Vehicle(0)
        vehicle.add_plate("29A12345", .95)
        for index, sample in enumerate((photo(.9, plate="30B88888"), photo(.9, ocr=.7), photo(.4), photo(.7), photo(.8))):
            sample["_frame_idx"] = index * 5
            vehicle.add_sample("plate", sample)
        row = vehicle.row("camera", 1, min_image_quality=.5, min_detection=.6, min_ocr=.85)
        self.assertEqual(len(row["Danh sách ảnh biển"]), 2)
        self.assertEqual(row["Ảnh biển số"]["quality_score"], .8)

    def test_low_quality_is_not_written_and_better_duplicate_updates_one_event(self):
        with TemporaryDirectory() as folder:
            store = EventStoreStub()
            recorder = EvidenceRecorder(store, folder)
            recorder.record(event(.3))
            self.assertEqual(store.recent(), [])
            self.assertEqual(list(Path(folder).rglob("*.jpg")), [])
            recorder.record(event(.7))
            first = store.recent()[0]
            recorder.record(event(.6, track=2))
            self.assertEqual(len(store.recent()), 1)
            recorder.record(event(.9, track=3, plate="29A-123.45"))
            rows = store.recent()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["event_id"], first["event_id"])
            self.assertEqual(rows[0]["images"]["plate"][0]["quality_score"], .9)
            self.assertEqual(rows[0]["images"]["detected"], first["images"]["detected"])
            self.assertEqual(len(list(Path(folder).rglob("*.jpg"))), 2)

    def test_different_camera_plate_or_run_is_not_suppressed(self):
        with TemporaryDirectory() as folder:
            store = EventStoreStub()
            recorder = EvidenceRecorder(store, folder)
            for row in (event(), event(camera="other"), event(plate="30B88888")):
                recorder.record(row)
            self.assertEqual(len(store.recent()), 3)
            EvidenceRecorder(store, folder).record(event())
            self.assertEqual(len(store.recent()), 4)
            self.assertEqual(len(unique_results(store.recent())), 3)

    def test_high_quality_does_not_override_rejected_ocr_or_detection(self):
        with TemporaryDirectory() as folder:
            store = EventStoreStub()
            recorder = EvidenceRecorder(store, folder)
            for bad in (photo(.99, ocr=.4), photo(.99, detection=.2), photo(.99, plate="30B88888")):
                row = event()
                row["Danh sách ảnh biển"] = [bad]
                recorder.record(row)
            self.assertEqual(store.recent(), [])


class CandidateConsensusTests(unittest.TestCase):
    def test_quality_penalizes_small_blurred_cut_or_distorted_plates(self):
        rng = np.random.default_rng(7)
        clear = rng.integers(0, 255, (32, 144, 3), dtype=np.uint8)

        def score(image, confidence=.95, box=(20, 20, 164, 52)):
            return plate_quality(image, confidence, box, (200, 300, 3))["quality_score"]

        baseline = score(clear)
        self.assertGreater(baseline, .8)
        self.assertLess(score(clear, .5), baseline)
        self.assertLess(score(cv2.resize(clear, (36, 8))), baseline)
        self.assertLess(score(np.full_like(clear, 100)), baseline)
        self.assertEqual(score(clear, box=(0, 20, 144, 52)), 0)
        self.assertLess(score(cv2.resize(clear, (320, 16))), baseline)

    def test_weighted_votes_and_margin_prevent_early_ambiguous_confirmation(self):
        vehicle = Vehicle(0)
        for text, score, count in (("29A12345", .95, 4), ("29A12346", .7, 5)):
            for _ in range(count):
                vehicle.add_plate(text, score)
        self.assertEqual(max(vehicle.plates, key=vehicle.plate_rank), "29A12345")
        vehicle.confirm_plate(3, 3, .85, 1)
        self.assertEqual(vehicle.confirmed_plate, "")
        vehicle.add_plate("29A12345", .95)
        vehicle.confirm_plate(3, 3, .85, 1)
        self.assertEqual(vehicle.confirmed_plate, "29A12345")

    def test_logo_votes_overcome_single_high_confidence_outlier(self):
        vehicle = Vehicle(0)
        vehicle.add_brand("wrong", .99, 1)
        for _ in range(3):
            vehicle.add_brand("correct", .8, .9)
        self.assertEqual(vehicle.brand, "correct")
        self.assertAlmostEqual(vehicle.brand_score, .8)

    def test_track_keeps_top_five_distinct_frames(self):
        vehicle = Vehicle(0, candidate_limit=5)
        for index in range(12):
            sample = photo(index / 12)
            sample["_frame_idx"] = index
            vehicle.add_sample("plate", sample)
        self.assertEqual(len(vehicle.plate_samples), 5)
        self.assertEqual({x["_frame_idx"] for x in vehicle.plate_samples}, {7, 8, 9, 10, 11})
        row = vehicle.row("cam", 1, min_image_quality=.9, keep_low_confidence=True, top_k=5)
        self.assertEqual(row["confidence_status"], "low")
        self.assertEqual(len(row["Danh sách ảnh biển"]), 5)

    def test_rejection_reasons_survive_row_and_include_failed_gates(self):
        vehicle = Vehicle(0)
        vehicle.reject("vehicle_too_small", "80px < 120px")
        vehicle.add_plate("29A12345", .9)
        row = vehicle.row("cam", 1, min_image_quality=.9, keep_low_confidence=True, top_k=3)
        self.assertEqual(row["rejection_reasons"]["vehicle_too_small"], 1)
        self.assertEqual(row["rejection_detail"]["vehicle_too_small"], "80px < 120px")
        self.assertIn("quality_below_threshold", row["rejection_reasons"])
        self.assertIn("plate_not_confirmed", row["rejection_reasons"])

    def test_unread_fallback_keeps_best_image_and_stays_separate_from_other_tracks(self):
        with TemporaryDirectory() as folder:
            store = EventStoreStub()
            recorder = EvidenceRecorder(store, folder, keep_low_confidence=True)
            for track in (1, 2):
                row = event(.2, track=track, plate="")
                row["confidence_status"] = "low"
                recorder.record(row)
            rows = unique_results(store.recent())
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(row["confidence_status"] == "low" for row in rows))
            self.assertTrue(all(row["images"]["plate"] for row in rows))

    def test_confirmed_result_promotes_fallback_without_duplicate_and_prunes_images(self):
        with TemporaryDirectory() as folder:
            store = EventStoreStub()
            recorder = EvidenceRecorder(store, folder, keep_low_confidence=True, top_k=3)
            low = event(.2)
            low["confidence_status"] = "low"
            recorder.record(low)
            event_id = store.recent()[0]["event_id"]
            high = event(.9, track=2)
            high["confidence_status"] = "confirmed"
            candidates = []
            for index in range(5):
                sample = photo(.7 + index * .05)
                sample["data"] = b64encode(f"candidate-{index}".encode()).decode()
                candidates.append(sample)
            high["Danh sách ảnh biển"] = candidates
            recorder.record(high)
            rows = unique_results(store.recent())
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["event_id"], event_id)
            self.assertEqual(rows[0]["confidence_status"], "confirmed")
            self.assertEqual(len(rows[0]["images"]["plate"]), 3)
            self.assertEqual(len(list(Path(folder).rglob("plate_*.jpg"))), 3)
            recorder.record(low)
            self.assertEqual(unique_results(store.recent())[0]["confidence_status"], "confirmed")


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


if __name__ == "__main__":
    unittest.main()
