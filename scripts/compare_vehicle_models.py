"""Compare COCO baseline with a custom 4-class detector on identical test pixels.

Only baseline ground-truth IDs are mapped back to COCO. No weights are modified.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import yaml
from ultralytics import YOLO

from vehicle_ai.datasets import sha256

CUSTOM_TO_COCO = {0: 2, 1: 3, 2: 5, 3: 7}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--baseline", default="models/yolo11n.pt")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    data_path = Path(args.data).resolve()
    data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    root = Path(data["path"])
    if not root.is_absolute():
        root = (data_path.parent / root).resolve()
    test_images = root / data["test"]
    test_labels = root / "labels" / "test"
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    baseline = YOLO(args.baseline)
    expected = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
    if any(baseline.names.get(k) != name for k, name in expected.items()):
        raise ValueError("Baseline must use the COCO vehicle class mapping.")
    candidate = YOLO(args.candidate)
    names = data["names"]
    if isinstance(names, list):
        names = dict(enumerate(names))
    if candidate.names != names or set(names) != set(CUSTOM_TO_COCO):
        raise ValueError("Candidate class names must match the 4-class dataset.")
    destination = output / "baseline_dataset"
    image_out, label_out = destination / "images/test", destination / "labels/test"
    image_out.mkdir(parents=True)
    label_out.mkdir(parents=True)
    counts = dict.fromkeys(range(4), 0)
    samples = []
    for image in sorted(test_images.iterdir()):
        if not image.is_file():
            continue
        label = test_labels / (image.stem + ".txt")
        lines = []
        for line in label.read_text(encoding="utf-8").splitlines():
            tokens = line.split()
            if not tokens:
                continue
            original = int(tokens[0])
            tokens[0] = str(CUSTOM_TO_COCO[original])
            lines.append(" ".join(tokens))
            counts[original] += 1
        shutil.copy2(image, image_out / image.name)
        (label_out / label.name).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        assert sha256(image) == sha256(image_out / image.name)
        # Verify each box remains identical after reversing the class remap.
        original_lines = [line.split() for line in label.read_text().splitlines() if line.strip()]
        remapped_lines = [line.split() for line in lines]
        assert all(a[1:] == b[1:] and CUSTOM_TO_COCO[int(a[0])] == int(b[0])
                   for a, b in zip(original_lines, remapped_lines))
        samples.append({"image": image.name, "image_sha256": sha256(image), "label_sha256": sha256(label)})
    if not samples:
        raise ValueError("Test split is empty.")
    mapped_yaml = destination / "dataset.yaml"
    mapped_yaml.write_text(yaml.safe_dump({
        "path": str(destination), "train": "images/test", "val": "images/test", "test": "images/test",
        "names": baseline.names,
    }, allow_unicode=True), encoding="utf-8")
    common = dict(split="test", imgsz=640, batch=4, device="cpu", workers=0,
                  conf=0.001, iou=0.7, max_det=300, rect=True,
                  augment=False, agnostic_nms=False, plots=True, save_json=False)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "settings": common, "class_mapping_custom_to_coco": CUSTOM_TO_COCO,
        "test_images": len(samples), "test_instances": sum(counts.values()),
        "class_counts": counts, "samples": samples,
        "note": "Dataset was model-assisted and human-reviewed; small test split from one video.",
        "models": {},
    }
    for name, model, yaml_file, classes, mapping, weights in (
        ("baseline", baseline, mapped_yaml, list(CUSTOM_TO_COCO.values()),
         CUSTOM_TO_COCO, args.baseline),
        ("candidate", candidate, data_path, list(CUSTOM_TO_COCO), {i: i for i in range(4)}, args.candidate),
    ):
        metrics = model.val(data=str(yaml_file), classes=classes, project=str(output), name=name, **common)
        indices = {int(cls): i for i, cls in enumerate(metrics.box.ap_class_index)}
        per_class = {}
        for cls, target in mapping.items():
            index = indices.get(target)
            per_class[names[cls]] = {
                "instances": counts[cls],
                "precision": float(metrics.box.p[index]) if index is not None else None,
                "recall": float(metrics.box.r[index]) if index is not None else None,
                "map50": float(metrics.box.ap50[index]) if index is not None else None,
                "map50_95": float(metrics.box.ap[index]) if index is not None else None,
            }
        report["models"][name] = {
            "weights": str(Path(weights).resolve()), "sha256": sha256(weights),
            "metrics": {k: float(v) for k, v in metrics.results_dict.items()},
            "per_class": per_class,
        }
        (output / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
