"""OCR adapters and plate text normalization."""
import cv2
import numpy as np
import onnxruntime as ort
import yaml

PLATE_CHARACTERS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")


def read_plate(recognizer, image):
    result = inspect_plate(recognizer, image)
    return (result["text"], result["score"]) if result["accepted"] else ("", 0.0)


def inspect_plate(recognizer, image):
    """Keep rejected OCR output visible for diagnosis without accepting it."""
    return inspect_predictions(recognizer.predict(input=image))


def inspect_plates(recognizer, images):
    if not images:
        return []
    batch = getattr(type(recognizer), "predict_batch", None)
    if batch is None:
        return [inspect_plate(recognizer, image) for image in images]
    results = recognizer.predict_batch(images)
    if len(results) != len(images):
        raise ValueError("OCR batch trả về sai số lượng kết quả.")
    return [inspect_predictions(result) for result in results]


def inspect_predictions(results):
    best = {"text": "", "score": 0.0, "accepted": False}
    for result in results:
        plate = "".join(c for c in result["rec_text"].upper() if c in PLATE_CHARACTERS)
        score = float(result["rec_score"])
        candidate = {"text": plate, "score": score, "accepted": 5 <= len(plate) <= 11 and score >= 0.65}
        if candidate["accepted"]:
            return candidate
        if score >= best["score"]:
            best = candidate
    return best


class VietnamesePlateRecognizer:
    def __init__(self, onnx_path, config_path, *, threads=4, batch_size=8):
        with open(config_path, encoding="utf-8") as file:
            config = yaml.safe_load(file)
        self.height = int(config.get("img_height", 64))
        self.width = int(config.get("img_width", 128))
        self.alphabet = str(config.get("alphabet", "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_"))
        self.pad_char = str(config.get("pad_char", "_"))
        providers = [
            provider for provider in ("CUDAExecutionProvider", "CPUExecutionProvider")
            if provider in ort.get_available_providers()
        ]
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(onnx_path), sess_options=options, providers=providers)
        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        self.fixed_batch = model_input.shape[0] if isinstance(model_input.shape[0], int) else None
        self.batch_size = self.fixed_batch or batch_size

    def predict(self, input):
        return self.predict_batch([input])[0]

    def predict_batch(self, images):
        results = []
        for start in range(0, len(images), self.batch_size):
            chunk = images[start:start + self.batch_size]
            tensors = [
                cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (self.width, self.height),
                           interpolation=cv2.INTER_LINEAR).astype(np.uint8)
                for image in chunk
            ]
            if self.fixed_batch:
                tensors.extend([tensors[-1]] * (self.fixed_batch - len(tensors)))
            probabilities = self.session.run(None, {self.input_name: np.stack(tensors)})[0]
            if len(probabilities) != len(tensors):
                raise ValueError("OCR model trả về sai kích thước batch.")
            results.extend(self.decode(item) for item in probabilities[:len(chunk)])
        return results

    def decode(self, probabilities):
        text, scores = [], []
        # CCT predicts character slots, not CTC timesteps: preserve repeated characters.
        for index, confidence in zip(probabilities.argmax(axis=-1), probabilities.max(axis=-1)):
            char = self.alphabet[int(index)] if int(index) < len(self.alphabet) else ""
            if char and char != self.pad_char:
                text.append(char)
                scores.append(float(confidence))
        return [{
            "rec_text": "".join(text),
            "rec_score": sum(scores) / len(scores) if scores else 0.0,
        }]
