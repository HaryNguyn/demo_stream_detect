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
    best = {"text": "", "score": 0.0, "accepted": False}
    for result in recognizer.predict(input=image):
        plate = "".join(c for c in result["rec_text"].upper() if c in PLATE_CHARACTERS)
        score = float(result["rec_score"])
        candidate = {"text": plate, "score": score, "accepted": 5 <= len(plate) <= 11 and score >= 0.65}
        if candidate["accepted"]:
            return candidate
        if score >= best["score"]:
            best = candidate
    return best


class VietnamesePlateRecognizer:
    def __init__(self, onnx_path, config_path):
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
        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, input):
        image = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        tensor = image.astype(np.uint8)[np.newaxis, ...]
        probabilities = self.session.run(None, {self.input_name: tensor})[0][0]
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
