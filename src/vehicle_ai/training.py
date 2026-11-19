"""Training entry points. Test split is never used for checkpoint selection."""
import json
import random
from pathlib import Path

import cv2
import numpy as np

from .datasets import ALPHABET, build_yolo, read_manifest, sha256, validate_manifest
from .evaluation import edit_distance


def train_detector(manifest, names, weights, output, *, epochs=50, batch=8, imgsz=640, device="cpu"):
    from ultralytics import YOLO

    if epochs <= 0 or batch <= 0 or imgsz <= 0:
        raise ValueError("epochs, batch, imgsz phải > 0.")
    validate_manifest(manifest, "detect", names)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    build_yolo(manifest, output / "dataset", names)
    metadata = {
        "task": "detect", "initial_weights": weights, "manifest_sha256": sha256(manifest),
        "names": names, "epochs": epochs, "batch": batch, "imgsz": imgsz, "device": device,
        "seed": 42,
    }
    (output / "training.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    model = YOLO(weights)
    model.train(data=str(output / "dataset" / "dataset.yaml"), epochs=epochs, batch=batch,
                imgsz=imgsz, device=device, seed=42, deterministic=True, workers=0,
                project=str(output), name="fit", exist_ok=False)
    return {"best_checkpoint": str(output / "fit" / "weights" / "best.pt")}


def train_crnn(manifest, output, *, epochs=50, batch=32, learning_rate=0.001, device="cpu"):
    import torch
    from torch.utils.data import DataLoader, Dataset
    from .crnn import CRNN, HEIGHT, WIDTH, decode, preprocess

    if epochs <= 0 or batch <= 0 or learning_rate <= 0:
        raise ValueError("epochs, batch, learning_rate phải > 0.")
    validate_manifest(manifest, "ocr")
    rows = read_manifest(manifest)
    device = f"cuda:{device}" if str(device).isdigit() else device
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    class Crops(Dataset):
        def __init__(self, split):
            self.rows = [row for row in rows if row["split"] == split]

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            row = self.rows[index]
            return torch.from_numpy(preprocess(cv2.imread(row["image"]))), row["text"]

    train = DataLoader(Crops("train"), batch_size=batch, shuffle=True, num_workers=0)
    validation = DataLoader(Crops("val"), batch_size=batch, shuffle=False, num_workers=0)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    model = CRNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=False)
    metadata = {
        "task": "ocr", "architecture": "vehicle_ai_crnn_v1", "alphabet": ALPHABET,
        "height": HEIGHT, "width": WIDTH, "manifest_sha256": sha256(manifest),
        "epochs": epochs, "batch": batch, "learning_rate": learning_rate, "seed": 42,
    }
    (output / "training.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    best_rank = (-1, -float("inf"))
    with (output / "history.jsonl").open("x", encoding="utf-8") as history:
        for epoch in range(1, epochs + 1):
            model.train()
            loss_sum = samples = 0
            for images, texts in train:
                targets = torch.tensor([ALPHABET.index(c) + 1 for text in texts for c in text], dtype=torch.long)
                target_lengths = torch.tensor([len(text) for text in texts], dtype=torch.long)
                optimizer.zero_grad()
                logits = model(images.to(device))
                required = [len(text) + sum(a == b for a, b in zip(text, text[1:])) for text in texts]
                if max(required) > logits.shape[0]:
                    raise ValueError("Nhãn dài hơn số timestep CTC.")
                lengths = torch.full((len(texts),), logits.shape[0], dtype=torch.long)
                loss = criterion(logits.log_softmax(-1), targets, lengths, target_lengths)
                if not torch.isfinite(loss):
                    raise RuntimeError("CTC loss không hữu hạn; kiểm tra nhãn và cấu hình.")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                loss_sum += loss.item() * len(texts)
                samples += len(texts)
            model.eval()
            correct = total = edits = characters = 0
            with torch.inference_mode():
                for images, texts in validation:
                    predictions = decode(model(images.to(device)).cpu())
                    for text, prediction in zip(texts, predictions):
                        correct += text == prediction["rec_text"]
                        total += 1
                        edits += edit_distance(text, prediction["rec_text"])
                        characters += len(text)
            metrics = {"epoch": epoch, "train_loss": loss_sum / samples,
                       "val_exact_match": correct / total, "val_cer": edits / characters}
            history.write(json.dumps(metrics) + "\n")
            history.flush()
            print(json.dumps(metrics), flush=True)
            checkpoint = {**metadata, "epoch": epoch,
                          "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                          "validation": metrics}
            torch.save(checkpoint, output / "last.pt")
            rank = (metrics["val_exact_match"], -metrics["val_cer"])
            if rank > best_rank:
                best_rank = rank
                torch.save(checkpoint, output / "best.pt")
    return {"best_checkpoint": str(output / "best.pt"), "last_validation": metrics}


def export_model(weights, engine, output=None, imgsz=640):
    if engine == "detector":
        from ultralytics import YOLO

        return {"onnx": str(YOLO(weights).export(format="onnx", imgsz=imgsz, dynamic=False, simplify=False))}
    import torch
    from .crnn import CRNNRecognizer, HEIGHT, WIDTH

    output = Path(output or str(Path(weights).with_suffix(".onnx")))
    if output.exists() or Path(str(output) + ".json").exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    recognizer = CRNNRecognizer(weights)
    sample = torch.zeros(1, 1, HEIGHT, WIDTH)
    torch.onnx.export(recognizer.model, sample, str(output), input_names=["images"],
                      output_names=["logits"], opset_version=17, dynamo=False)
    Path(str(output) + ".json").write_text(json.dumps({
        "architecture": "vehicle_ai_crnn_v1", "alphabet": recognizer.alphabet,
        "height": HEIGHT, "width": WIDTH, "input": "NCHW grayscale [-1,1], centered letterbox",
        "output": "TNC logits, CTC blank=0", "source_sha256": sha256(weights),
    }, indent=2), encoding="utf-8")
    return {"onnx": str(output)}


def evaluate_ocr(manifest, recognizer, split="test"):
    validate_manifest(manifest, "ocr")
    rows = [row for row in read_manifest(manifest) if row["split"] == split]
    correct = edits = characters = accepted = 0
    results = []
    from .ocr import inspect_plate

    for row in rows:
        prediction = inspect_plate(recognizer, cv2.imread(row["image"]))
        correct += row["text"] == prediction["text"]
        edits += edit_distance(row["text"], prediction["text"])
        characters += len(row["text"])
        accepted += prediction["accepted"]
        results.append({"image": row["image"], "truth": row["text"], **prediction})
    return {
        "split": split, "samples": len(rows), "exact_match": correct / len(rows),
        "cer": edits / characters, "acceptance_coverage": accepted / len(rows), "predictions": results,
    }
