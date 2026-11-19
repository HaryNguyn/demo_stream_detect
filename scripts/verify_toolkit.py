"""Integration smoke checks only, NOT an accuracy benchmark or a production model."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import ultralytics

from vehicle_ai.cli import main as cli
from vehicle_ai.crnn import CRNNRecognizer, preprocess
from vehicle_ai.datasets import write_manifest
from vehicle_ai.training import export_model, train_crnn, train_detector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--detector-training", action="store_true")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    rows = []
    for i, split in enumerate(("train", "val", "test")):
        image = np.full((64, 160, 3), 40 + i * 60, dtype=np.uint8)
        cv2.putText(image, "30A11111", (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (250, 250, 250), 1)
        image_path = output / f"synthetic_{i}.png"
        cv2.imwrite(str(image_path), image)
        label_path = output / f"synthetic_{i}.txt"
        label_path.write_text("0 0.5 0.5 0.9 0.8\n", encoding="utf-8")
        rows.append({"image": str(image_path), "label": str(label_path), "text": "30A11111",
                     "group_id": f"synthetic-{i}", "split": split})
    manifest = output / "synthetic.csv"
    write_manifest(manifest, rows)
    trained = train_crnn(manifest, output / "ocr-smoke", epochs=1, batch=1)
    exported = export_model(trained["best_checkpoint"], "crnn", str(output / "ocr-smoke.onnx"))
    pytorch_rec = CRNNRecognizer(trained["best_checkpoint"])
    onnx_rec = CRNNRecognizer(exported["onnx"])
    tensor = preprocess(cv2.imread(rows[0]["image"]))[None]
    with torch.inference_mode():
        expected = pytorch_rec.model(torch.from_numpy(tensor)).numpy()
    actual = onnx_rec.session.run(None, {onnx_rec.input_name: tensor})[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-5)

    sample_path = Path(ultralytics.__file__).parent / "assets" / "bus.jpg"
    image = cv2.imread(str(sample_path)) if sample_path.is_file() else np.zeros((480, 640, 3), np.uint8)
    image = cv2.resize(image, (640, 480))
    video = output / "sample.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (640, 480))
    if not writer.isOpened():
        raise RuntimeError("MJPG encoder unavailable.")
    for _ in range(5):
        writer.write(image)
    writer.release()
    cli(["run", "--source", str(video), "--source-id", "smoke-only", "--camera", "TEST",
         "--config", "configs/baseline.yaml", "--output", str(output / "inference")])
    cli(["extract", "--video", str(video), "--output", str(output / "extraction"),
         "--group-id", "smoke-only", "--every-seconds", "0.4"])
    summary = {
        "purpose": "integration smoke test only; not accuracy evaluation",
        "ocr_train": "passed_one_epoch_synthetic",
        "ocr_onnx_parity": "passed",
        "baseline_inference": "passed",
        "frame_extraction": "passed",
        "sample_source": "ultralytics bus.jpg" if sample_path.is_file() else "blank frames",
        "detector_train": "not_requested",
    }
    if args.detector_training:
        trained_detector = train_detector(
            manifest, ["plate"], str(Path("models/yolo11n.pt").resolve()),
            output / "detector-smoke", epochs=1, batch=1, imgsz=64, device="cpu",
        )
        export_model(trained_detector["best_checkpoint"], "detector", imgsz=64)
        summary["detector_train"] = "passed_one_epoch_synthetic"
        summary["detector_export"] = "passed"
    (output / "verification.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
