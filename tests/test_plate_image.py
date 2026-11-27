"""Still-image recognition page: isolated HTTP routes and core logic."""
import base64
from io import BytesIO
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from event_store_stub import EventStoreStub
from vehicle_ai.config import PipelineConfig
from vehicle_ai.plate_image import _full_image_plates, _ocr_plate, analyze_image, decode_image, normalize_options
from vehicle_ai.web import create_app
from vehicle_ai.plate_image import recognize_image


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
    def test_mixed_inputs_use_the_appropriate_route(self):
        models = {"recognizer": _Recognizer()}
        for kind, shape, expected in [
            ("auto", (48, 140), "crop"), ("crop", (720, 960), "crop"),
            ("auto", (720, 960), "scene"), ("scene", (48, 140), "scene"),
        ]:
            with (
                self.subTest(kind=kind, shape=shape),
                patch("vehicle_ai.plate_image.get_models", return_value=models) as load,
                patch("vehicle_ai.plate_image.analyze_image", return_value={"vehicles": [{}]}) as scene,
            ):
                _, payload = cv2.imencode(".png", np.zeros((*shape, 3), np.uint8))
                result = recognize_image(payload.tobytes(), {"input_kind": kind}, preview="none")
                self.assertEqual(result["input_route"], expected)
                self.assertEqual(scene.call_count, int(expected == "scene"))
                self.assertEqual(load.call_args.kwargs["ocr_only"], expected == "crop")

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

    def test_ocr_plate_avoids_fallback_for_clear_read(self):
        recognizer = Mock(spec=_Recognizer)
        recognizer.predict.return_value = [{"rec_text": "30A12345", "rec_score": 0.97}]
        result = _ocr_plate(recognizer, np.zeros((30, 120, 3), np.uint8))
        self.assertEqual((result["text"], result["variant"]), ("30A12345", "original"))
        recognizer.predict.assert_called_once()

    def test_ocr_plate_reads_two_lines_side_by_side(self):
        class TwoLineRecognizer:
            def predict_batch(self, images):
                return [[{"rec_text": "30A12345", "rec_score": 0.97}]
                        if image.shape[1] / image.shape[0] > 5 else []
                        for image in images]

        result = _ocr_plate(TwoLineRecognizer(), np.zeros((32, 48, 3), np.uint8))
        self.assertEqual((result["text"], result["variant"]), ("30A12345", "original_two_lines"))

    def test_ocr_plate_recovers_sideways_crop(self):
        class LandscapeRecognizer:
            def predict_batch(self, images):
                return [[{"rec_text": "30A12345", "rec_score": 0.97}]
                        if image.shape[1] / image.shape[0] > 2 else []
                        for image in images]

        result = _ocr_plate(LandscapeRecognizer(), np.zeros((90, 30, 3), np.uint8))
        self.assertEqual((result["text"], result["variant"]), ("30A12345", "rotate_90"))

    def test_portrait_detection_maps_rotated_plate_to_original_coordinates(self):
        detector = Mock()
        detector.predict.side_effect = lambda image, **kwargs: [
            _Result([_Box([40, 20, 100, 40], 0.95, 0)] if image.shape[1] > image.shape[0] else [])
        ]
        found = _full_image_plates(detector, np.zeros((200, 100, 3), np.uint8), normalize_options({}))
        self.assertEqual(found, [((20, 100, 40, 160), 0.95)])

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
        self.assertEqual(result["summary"], {"vehicles": 0, "plates_detected": 1, "plates_accepted": 1})
        vehicle = result["vehicles"][0]
        self.assertEqual(vehicle["vehicle_type"], "")
        self.assertEqual(vehicle["brand"], "")
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

        self.assertEqual(result["summary"], {"vehicles": 0, "plates_detected": 1, "plates_accepted": 1})
        self.assertEqual(result["vehicles"][0]["vehicle_type"], "")
        self.assertEqual(plate.calls, 2)

    def test_analyze_image_recovers_plate_when_vehicle_detector_misses(self):
        config = PipelineConfig(vehicle_weights="models/yolo11n.pt")
        models = {
            "config": config, "detector": _EmptyDetector(), "logo": None,
            "plate": _Plate(), "recognizer": _Recognizer(),
        }

        result = analyze_image(np.zeros((240, 240, 3), np.uint8), models, normalize_options({}))

        self.assertEqual(result["summary"], {"vehicles": 0, "plates_detected": 1, "plates_accepted": 1})
        vehicle = result["vehicles"][0]
        self.assertEqual(vehicle["vehicle_type"], "")
        self.assertEqual(vehicle["plate_text"], "29A12345")

    def test_analyze_image_accepts_person_candidate_only_with_plate_below(self):
        config = PipelineConfig(vehicle_weights="models/yolo11n.pt")
        models = {
            "config": config, "detector": _RiderDetector(), "logo": None,
            "plate": _RiderPlate(), "recognizer": _Recognizer(),
        }

        result = analyze_image(np.zeros((240, 240, 3), np.uint8), models, normalize_options({}))

        self.assertEqual(result["summary"], {"vehicles": 0, "plates_detected": 1, "plates_accepted": 1})
        vehicle = result["vehicles"][0]
        self.assertEqual(vehicle["vehicle_type"], "")
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
    def test_stream_delivers_first_result_before_processing_second_image(self):
        with patch("vehicle_ai.plate_image.recognize_image", side_effect=lambda *_: {}) as recognize:
            response = self.client.post("/api/plate/recognize", headers={"Accept": "application/x-ndjson"},
                                        data={"image": [(BytesIO(b"a"), "a.jpg"), (BytesIO(b"b"), "b.jpg")]},
                                        buffered=False)
            with response:
                chunks = iter(response.response)
                self.assertIn(b"a.jpg", next(chunks))
                self.assertEqual(recognize.call_count, 1)
                self.assertIn(b"b.jpg", next(chunks))
                self.assertEqual([call.args[0] for call in recognize.call_args_list], [b"a", b"b"])

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
        self.assertIn("plate_weights", config["available"])
        self.assertNotIn("vehicle_weights", config["available"])

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

    def test_image_endpoint_requests_crop_only_when_selected(self):
        with patch("vehicle_ai.plate_image.recognize_image", return_value={}) as recognizer:
            response = self.client.post("/api/plate/recognize", data={
                "image": (BytesIO(b"image"), "car.jpg"),
                "preview": "crop",
            })
        self.assertEqual(response.status_code, 200)
        recognizer.assert_called_once_with(b"image", {}, preview="crop")


if __name__ == "__main__":
    unittest.main()
