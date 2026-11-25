"""Still-image recognition page: isolated HTTP routes and core logic."""
import base64
from io import BytesIO
from pathlib import Path
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from event_store_stub import EventStoreStub
from vehicle_ai.config import PipelineConfig
from vehicle_ai.plate_image import _ocr_plate, analyze_image, decode_image, normalize_options
from vehicle_ai.web import create_app


class _Scalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class _Tensor:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


class _Box:
    def __init__(self, xyxy, confidence, class_id):
        self.xyxy = [_Tensor(xyxy)]
        self.conf = _Scalar(confidence)
        self.cls = _Scalar(class_id)


class _Result:
    def __init__(self, boxes, names=None):
        self.boxes = boxes
        self.names = names or {}


class _Detector:
    names = {2: "Ô tô con"}

    def predict(self, image, **kwargs):
        return [_Result([_Box([20, 20, 180, 140], 0.9, 2)], self.names)]


class _EmptyDetector:
    names = {2: "Ô tô con"}

    def predict(self, image, **kwargs):
        return [_Result([])]


class _RiderDetector:
    names = {0: "person"}

    def predict(self, image, **kwargs):
        return [_Result([_Box([20, 20, 180, 140], 0.9, 0)], self.names)]


class _MixedDetector:
    names = {0: "person", 1: "bicycle", 2: "Ô tô con", 3: "Xe máy"}

    def predict(self, image, **kwargs):
        boxes = [
            _Box([5, 5, 40, 100], 0.95, 0),
            _Box([45, 5, 90, 100], 0.9, 1),
            _Box([95, 20, 180, 140], 0.9, 2),
            _Box([185, 20, 235, 140], 0.9, 3),
        ]
        return [_Result(boxes, self.names)]


class _Logo:
    names = {0: "Toyota"}

    def predict(self, image, **kwargs):
        return [_Result([_Box([10, 10, 60, 60], 0.8, 0)], self.names)]


class _Plate:
    def predict(self, image, **kwargs):
        return [_Result([_Box([10, 80, 150, 120], 0.9, 0)])]


class _SecondPlateOnly:
    def __init__(self):
        self.calls = 0

    def predict(self, image, **kwargs):
        self.calls += 1
        boxes = [_Box([10, 80, 150, 120], 0.9, 0)] if self.calls == 2 else []
        return [_Result(boxes)]


class _RiderPlate:
    def predict(self, image, **kwargs):
        return [_Result([_Box([120, 140, 190, 170], 0.9, 0)])]


class _NoPlate:
    def predict(self, image, **kwargs):
        return [_Result([])]


class _Recognizer:
    def predict(self, input):
        return [{"rec_text": "29A12345", "rec_score": 0.95}]


class _VariantRecognizer:
    def predict_batch(self, images):
        return [[{"rec_text": "29A12345" if image.shape[1] > 140 else "29A12346",
                  "rec_score": 0.99 if image.shape[1] > 140 else 0.8}]
                for image in images]


class PlateImageTests(unittest.TestCase):
    def test_normalize_options_rejects_bad_numbers(self):
        self.assertEqual(normalize_options({})["mode"], "manual")
        with self.assertRaises(ValueError):
            normalize_options({"vehicle_conf": "1.5"})
        with self.assertRaises(ValueError):
            normalize_options({"detector_imgsz": "0"})

    def test_ocr_plate_prefers_best_variant(self):
        result = _ocr_plate(_VariantRecognizer(), np.zeros((40, 140, 3), np.uint8))
        self.assertTrue(result["accepted"])
        self.assertEqual(result["text"], "29A12345")
        self.assertAlmostEqual(result["score"], 0.99)

    def test_decode_image_caps_large_inputs(self):
        image = np.zeros((3000, 4000, 3), np.uint8)
        ok, encoded = cv2.imencode(".jpg", image)
        self.assertTrue(ok)
        decoded = decode_image(encoded.tobytes())
        self.assertEqual(max(decoded.shape[:2]), 3000)
        with self.assertRaises(ValueError):
            decode_image(b"not-an-image")

    def test_analyze_image_reports_classification_and_plate(self):
        config = PipelineConfig(vehicle_weights="models/yolo11n.pt")
        models = {
            "config": config, "detector": _Detector(), "logo": _Logo(),
            "plate": _Plate(), "recognizer": _Recognizer(),
        }
        result = analyze_image(np.zeros((200, 240, 3), np.uint8), models, normalize_options({}))
        self.assertEqual(result["summary"], {"vehicles": 1, "plates_detected": 1, "plates_accepted": 1})
        vehicle = result["vehicles"][0]
        self.assertEqual(vehicle["vehicle_type"], "Ô tô con")
        self.assertEqual(vehicle["brand"], "Toyota")
        self.assertEqual(vehicle["plate_text"], "29A12345")
        self.assertTrue(vehicle["accepted"])
        self.assertEqual(vehicle["plate_bbox"], [10, 80, 150, 120])
        self.assertTrue(base64.b64decode(result["annotated"]).startswith(b"\xff\xd8"))

    def test_analyze_image_returns_only_configured_vehicles_with_a_plate(self):
        config = PipelineConfig(vehicle_weights="models/yolo11n.pt")
        plate = _SecondPlateOnly()
        models = {
            "config": config, "detector": _MixedDetector(), "logo": None,
            "plate": plate, "recognizer": _Recognizer(),
        }

        result = analyze_image(np.zeros((200, 240, 3), np.uint8), models, normalize_options({}))

        self.assertEqual(result["summary"], {"vehicles": 1, "plates_detected": 1, "plates_accepted": 1})
        self.assertEqual([vehicle["vehicle_type"] for vehicle in result["vehicles"]], ["Ô tô con"])
        self.assertGreaterEqual(plate.calls, 3)

    def test_analyze_image_recovers_plate_when_vehicle_detector_misses(self):
        config = PipelineConfig(vehicle_weights="models/yolo11n.pt")
        models = {
            "config": config, "detector": _EmptyDetector(), "logo": None,
            "plate": _Plate(), "recognizer": _Recognizer(),
        }

        result = analyze_image(np.zeros((240, 240, 3), np.uint8), models, normalize_options({}))

        self.assertEqual(result["summary"], {"vehicles": 1, "plates_detected": 1, "plates_accepted": 1})
        vehicle = result["vehicles"][0]
        self.assertTrue(vehicle["fallback"])
        self.assertEqual(vehicle["vehicle_type"], "Xe máy (suy luận)")
        self.assertEqual(vehicle["plate_text"], "29A12345")

    def test_analyze_image_accepts_person_candidate_only_with_plate_below(self):
        config = PipelineConfig(vehicle_weights="models/yolo11n.pt")
        models = {
            "config": config, "detector": _RiderDetector(), "logo": None,
            "plate": _RiderPlate(), "recognizer": _Recognizer(),
        }

        result = analyze_image(np.zeros((240, 240, 3), np.uint8), models, normalize_options({}))

        self.assertEqual(result["summary"], {"vehicles": 1, "plates_detected": 1, "plates_accepted": 1})
        vehicle = result["vehicles"][0]
        self.assertEqual(vehicle["vehicle_type"], "Xe máy (suy luận)")
        self.assertEqual(vehicle["plate_text"], "29A12345")
        self.assertGreaterEqual(vehicle["plate_bbox"][1], 140)

    def test_analyze_image_does_not_accept_person_without_plate(self):
        config = PipelineConfig(vehicle_weights="models/yolo11n.pt")
        models = {
            "config": config, "detector": _RiderDetector(), "logo": None,
            "plate": _NoPlate(), "recognizer": _Recognizer(),
        }

        result = analyze_image(np.zeros((240, 240, 3), np.uint8), models, normalize_options({}))

        self.assertEqual(result["summary"], {"vehicles": 0, "plates_detected": 0, "plates_accepted": 0})


class PlateWebTests(unittest.TestCase):
    def setUp(self):
        self.store = EventStoreStub()
        self.client = create_app(self.store).test_client()

    def test_page_and_config_are_available(self):
        response = self.client.get("/plate")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"/static/plate.js", response.data)
        self.assertIn(b"webkitdirectory", response.data)
        self.assertIn(b'id="queue-items"', response.data)
        self.assertIn(b'id="stop-recognition"', response.data)
        config = self.client.get("/api/plate/config").json
        self.assertEqual(config["defaults"]["plate_weights"], "models/best_stripped.onnx")
        self.assertIn("vehicle_weights", config["available"])

    def test_image_endpoint_does_not_touch_video_session_or_store(self):
        payload = {"results": [{"filename": "car.jpg", "vehicles": [], "annotated": "",
                                "summary": {"vehicles": 0, "plates_detected": 0, "plates_accepted": 0}}]}
        with patch("vehicle_ai.plate_image.recognize_image", return_value=payload["results"][0]) as recognizer:
            response = self.client.post("/api/plate/recognize", data={
                "image": (BytesIO(b"image-bytes"), "car.jpg"),
            })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["results"][0]["filename"], "car.jpg")
        self.assertEqual(recognizer.call_args.args[0], b"image-bytes")
        self.assertEqual(self.client.get("/api/status").json["status"], "idle")
        self.assertEqual(self.store.recent(), [])
        self.assertEqual(self.client.get("/api/events").json, [])

    def test_image_endpoint_accepts_multiple_files_with_folder_paths(self):
        def fake_recognize(payload, options):
            return {"payload": payload.decode(), "vehicles": [], "summary": {
                "vehicles": 0, "plates_detected": 0, "plates_accepted": 0,
            }}

        with patch("vehicle_ai.plate_image.recognize_image", side_effect=fake_recognize) as recognizer:
            response = self.client.post("/api/plate/recognize", data={
                "image": [
                    (BytesIO(b"first"), "dataset/first.jpg"),
                    (BytesIO(b"second"), "dataset/subfolder/second.png"),
                ],
            })

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            [result["filename"] for result in response.json["results"]],
            ["dataset/first.jpg", "dataset/subfolder/second.png"],
        )
        self.assertEqual(recognizer.call_count, 2)

    def test_invalid_upload_is_rejected_without_inference(self):
        with patch("vehicle_ai.plate_image.recognize_image") as recognizer:
            response = self.client.post("/api/plate/recognize", data={
                "image": (BytesIO(b"video"), "clip.mp4"),
            })
        self.assertEqual(response.status_code, 400)
        recognizer.assert_not_called()
        self.assertEqual(self.client.post("/api/plate/recognize", data={}).status_code, 400)


if __name__ == "__main__":
    unittest.main()
