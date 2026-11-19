"""Small trainable CRNN/CTC baseline, separate from the bundled CCT model."""
import cv2
import numpy as np
import torch
from torch import nn

from .datasets import ALPHABET

HEIGHT, WIDTH = 32, 160


def preprocess(image):
    if image is None or image.size == 0:
        raise ValueError("Crop OCR rỗng.")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    scale = min(WIDTH / gray.shape[1], HEIGHT / gray.shape[0])
    width, height = max(1, round(gray.shape[1] * scale)), max(1, round(gray.shape[0] * scale))
    resized = cv2.resize(gray, (width, height), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((HEIGHT, WIDTH), 127, dtype=np.uint8)
    y, x = (HEIGHT - height) // 2, (WIDTH - width) // 2
    canvas[y:y + height, x:x + width] = resized
    return (canvas.astype(np.float32)[None] / 127.5) - 1.0


class CRNN(nn.Module):
    def __init__(self, alphabet=ALPHABET):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d((2, 1)),
            nn.Conv2d(128, 128, (4, 1)), nn.ReLU(),
        )
        self.sequence = nn.LSTM(128, 128, num_layers=2, bidirectional=True)
        self.classifier = nn.Linear(256, len(alphabet) + 1)

    def forward(self, images):
        features = self.cnn(images).squeeze(2).permute(2, 0, 1)
        sequence, _ = self.sequence(features)
        return self.classifier(sequence)


def decode(logits, alphabet=ALPHABET):
    probabilities = torch.as_tensor(logits).softmax(-1)
    scores, tokens = probabilities.max(-1)
    results = []
    for column in range(tokens.shape[1]):
        chars, confidences, previous = [], [], 0
        for token, score in zip(tokens[:, column].tolist(), scores[:, column].tolist()):
            if token != 0 and token != previous:
                chars.append(alphabet[token - 1])
                confidences.append(score)
            previous = token
        results.append({
            "rec_text": "".join(chars),
            "rec_score": sum(confidences) / len(confidences) if confidences else 0.0,
        })
    return results


class CRNNRecognizer:
    def __init__(self, path, device="cpu"):
        self.device = torch.device(f"cuda:{device}" if str(device).isdigit() else device)
        self.onnx = str(path).lower().endswith(".onnx")
        if self.onnx:
            import json
            from pathlib import Path
            import onnxruntime as ort

            config = json.loads(Path(str(path) + ".json").read_text(encoding="utf-8"))
            self.alphabet = config["alphabet"]
            self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
        else:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            if checkpoint.get("architecture") != "vehicle_ai_crnn_v1":
                raise ValueError("Checkpoint không phải CRNN của vehicle-ai.")
            self.alphabet = checkpoint["alphabet"]
            self.model = CRNN(self.alphabet).to(self.device)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.model.eval()

    def predict(self, input):
        tensor = preprocess(input)[None]
        if self.onnx:
            logits = self.session.run(None, {self.input_name: tensor})[0]
        else:
            with torch.inference_mode():
                logits = self.model(torch.from_numpy(tensor).to(self.device)).cpu()
        return decode(logits, self.alphabet)
