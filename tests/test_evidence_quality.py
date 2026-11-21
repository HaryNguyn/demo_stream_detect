from base64 import b64encode
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from event_store_stub import EventStoreStub
from vehicle_ai.aggregation import Vehicle
from vehicle_ai.result_filter import EvidenceRecorder, unique_results


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
