import unittest
from base64 import b64encode
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from event_store_stub import EventStoreStub
from test_evidence_quality import event, photo
from vehicle_ai.aggregation import Vehicle
from vehicle_ai.quality import plate_quality
from vehicle_ai.result_filter import EvidenceRecorder, unique_results


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
