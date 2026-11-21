"""Reproducible data -> baseline -> train -> evaluation command line."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time
from importlib.metadata import version

import yaml
from .model_loader import load_recognizer


def names_from(path):
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    names = data["names"]
    if isinstance(names, dict):
        indexed = {int(key): value for key, value in names.items()}
        if sorted(indexed) != list(range(len(indexed))):
            raise ValueError("names phải đánh số liên tục từ 0.")
        names = [indexed[i] for i in range(len(indexed))]
    if not isinstance(names, list) or not names or len(set(names)) != len(names):
        raise ValueError("names cần danh sách tên lớp duy nhất.")
    return names


def run_experiment(args):
    from . import __version__
    from .config import load_config
    from .datasets import sha256
    from .inference import optional_yolo
    from .pipeline import analyze_video
    from .storage import ExperimentWriter, MergedExperimentStore
    from .result_filter import EvidenceRecorder

    config = load_config(args.config)
    if args.roi_file:
        from .roi import validate_roi
        roi_data = json.loads(Path(args.roi_file).read_text(encoding="utf-8"))
        config.roi = validate_roi(roi_data["points"])
    model_paths = {key: value for key, value in asdict(config).items()
                   if key.endswith("_weights") and value}
    hashes = {key: sha256(path) for key, path in model_paths.items()}
    if config.ocr_config:
        hashes["ocr_config"] = sha256(config.ocr_config)
    source = Path(args.source)
    metadata = {
        "tool_version": __version__, "python": platform.python_version(),
        "dependencies": {name: version(name) for name in (
            "ultralytics", "torch", "onnxruntime", "opencv-python", "numpy", "scipy",
        )},
        "created_at": datetime.now(timezone.utc).isoformat(), "config": asdict(config),
        "model_sha256": hashes, "source_id": args.source_id,
        "source_sha256": sha256(source) if source.is_file() else None,
        "camera_id": args.camera, "max_seconds": args.max_seconds,
        "status": "running",
        "realtime": getattr(args, "realtime", False),
    }
    writer = ExperimentWriter(args.output)
    merged = MergedExperimentStore(writer.directory, args.source_id, config.model_version)
    recorder = EvidenceRecorder(
        merged, writer.directory / "merged-evidence", config=config,
        min_quality=config.min_image_quality, min_detection=config.min_evidence_detection,
        min_ocr=config.min_evidence_ocr, top_k=config.evidence_top_k,
        keep_low_confidence=config.keep_low_confidence,
    )

    def event_ready(row):
        writer.append(row)
        recorder.record(row)

    manifest = writer.directory / "run.json"
    manifest.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    started = time.perf_counter()
    try:
        logo = optional_yolo(config.logo_weights)
        metadata["brand_classes"] = list(logo.names.values()) if logo is not None else []
        metadata["vehicle_classes"] = config.vehicle_classes
        manifest.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        analyze_video(
            args.source, args.camera, config.vehicle_weights,
            logo, optional_yolo(config.plate_weights),
            load_recognizer(config), on_event=event_ready, max_seconds=args.max_seconds,
            collect_results=False, config=config, source_id=args.source_id,
            on_stats=lambda stats: metadata.update(stats),
            realtime=getattr(args, "realtime", False),
        )
        metadata["status"] = "completed"
    except Exception as error:
        metadata["status"] = "failed"
        metadata["error_type"] = type(error).__name__
        raise
    finally:
        writer.close()
        merged.close()
        metadata["elapsed_seconds"] = time.perf_counter() - started
        metadata["events"] = writer.count
        metadata["merged_events"] = len(merged.rows)
        metadata["event_files"] = {"raw": "events.jsonl", "merged": "events.merged.jsonl"}
        metadata["throughput_fps_including_load"] = metadata.get("frames", 0) / metadata["elapsed_seconds"]
        manifest.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"output": str(writer.directory), **metadata}


def build_parser():
    parser = argparse.ArgumentParser(prog="vehicle-ai")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Run a reproducible baseline/candidate experiment.")
    run.add_argument("--config", default="configs/baseline.yaml")
    run.add_argument("--roi-file", help="JSON ROI saved from the review UI; overrides YAML roi.")
    run.add_argument("--source", required=True)
    run.add_argument("--source-id", required=True, help="Stable ID also used in ground truth.")
    run.add_argument("--camera", default="CAM-01")
    run.add_argument("--max-seconds", type=float, default=0)
    run.add_argument("--realtime", action="store_true", help="Use latest-frame skipping, like the web pipeline.")
    run.add_argument("--output", required=True, help="New directory; existing results are never overwritten.")
    extract = sub.add_parser("extract")
    extract.add_argument("--video", required=True)
    extract.add_argument("--output", required=True)
    extract.add_argument("--group-id", required=True)
    extract.add_argument("--every-seconds", type=float, default=1)
    extract.add_argument("--max-frames", type=int, default=300)
    split = sub.add_parser("split")
    split.add_argument("--manifest", required=True)
    split.add_argument("--output", required=True)
    split.add_argument("--seed", type=int, default=42)
    for command in ("validate-data", "build-yolo", "train-detector", "train-ocr"):
        item = sub.add_parser(command)
        item.add_argument("--manifest", required=True)
        if command == "validate-data":
            item.add_argument("--task", choices=["detect", "ocr"], required=True)
        if command != "train-ocr":
            item.add_argument("--classes", required=command != "validate-data")
        if command != "validate-data":
            item.add_argument("--output", required=True)
        if command.startswith("train"):
            item.add_argument("--epochs", type=int, default=50)
            item.add_argument("--batch", type=int, default=8)
            item.add_argument("--device", default="cpu")
        if command == "train-detector":
            item.add_argument("--weights", required=True)
            item.add_argument("--imgsz", type=int, default=640)
        if command == "train-ocr":
            item.add_argument("--learning-rate", type=float, default=0.001)
    export = sub.add_parser("export")
    export.add_argument("--weights", required=True)
    export.add_argument("--engine", choices=["detector", "crnn"], required=True)
    export.add_argument("--output", help="CRNN only; detector exports alongside its weights.")
    export.add_argument("--imgsz", type=int, default=640)
    evaluate = sub.add_parser("evaluate-events")
    evaluate.add_argument("--truth", required=True)
    evaluate.add_argument("--predictions", required=True)
    evaluate.add_argument("--baseline", help="Previous predictions on the same ground truth; report after-minus-before deltas.")
    evaluate.add_argument("--brand-classes", help="Actual model names YAML; otherwise read brand_classes from adjacent run.json.")
    evaluate.add_argument("--iou", type=float, default=0.3)
    evaluate.add_argument("--output", required=True)
    ocr = sub.add_parser("evaluate-ocr")
    ocr.add_argument("--manifest", required=True)
    ocr.add_argument("--config", required=True)
    ocr.add_argument("--split", choices=["val", "test"], default="test")
    ocr.add_argument("--output", required=True)
    detector = sub.add_parser("evaluate-detector")
    detector.add_argument("--weights", required=True)
    detector.add_argument("--data", required=True)
    detector.add_argument("--device", default="cpu")
    detector.add_argument("--output", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    from . import datasets

    if args.command == "run":
        result = run_experiment(args)
    elif args.command == "extract":
        result = datasets.extract_frames(args.video, args.output, args.group_id, args.every_seconds, args.max_frames)
    elif args.command == "split":
        result = datasets.split_manifest(args.manifest, args.output, args.seed)
    elif args.command == "validate-data":
        result = datasets.validate_manifest(args.manifest, args.task, names_from(args.classes) if args.classes else None)
    elif args.command == "build-yolo":
        result = datasets.build_yolo(args.manifest, args.output, names_from(args.classes))
    elif args.command in {"train-detector", "train-ocr", "export", "evaluate-ocr"}:
        from . import training
        if args.command == "train-detector":
            result = training.train_detector(
                args.manifest, names_from(args.classes), args.weights, args.output,
                epochs=args.epochs, batch=args.batch, imgsz=args.imgsz, device=args.device,
            )
        elif args.command == "train-ocr":
            result = training.train_crnn(
                args.manifest, args.output, epochs=args.epochs, batch=args.batch,
                learning_rate=args.learning_rate, device=args.device,
            )
        elif args.command == "export":
            if args.engine == "detector" and args.output:
                raise ValueError("Detector export đặt ONNX cạnh weights; không dùng --output.")
            result = training.export_model(args.weights, args.engine, args.output, args.imgsz)
        else:
            from .config import load_config
            recognizer = load_recognizer(load_config(args.config))
            if recognizer is None:
                raise ValueError("Cấu hình chưa bật OCR.")
            result = training.evaluate_ocr(args.manifest, recognizer, args.split)
    elif args.command == "evaluate-events":
        from .evaluation import compare_reports, evaluate_events, read_events
        truth = read_events(args.truth)
        def brand_classes(predictions):
            if args.brand_classes:
                return names_from(args.brand_classes)
            manifest = Path(predictions).parent / "run.json"
            return json.loads(manifest.read_text(encoding="utf-8")).get("brand_classes") if manifest.is_file() else None

        result = evaluate_events(truth, read_events(args.predictions), args.iou,
                                 known_brands=brand_classes(args.predictions))
        if args.baseline:
            baseline = evaluate_events(truth, read_events(args.baseline), args.iou,
                                       known_brands=brand_classes(args.baseline))
            result["delta_from_baseline"] = compare_reports(baseline, result)
            result["baseline"] = baseline
    elif args.command == "evaluate-detector":
        from ultralytics import YOLO
        output = Path(args.output).resolve()
        if output.exists():
            raise FileExistsError(output)
        metrics = YOLO(args.weights).val(data=str(Path(args.data).resolve()), split="test",
                                        device=args.device, project=str(output.parent), name=output.name)
        result = {key: float(value) for key, value in metrics.results_dict.items()}
        output.mkdir(parents=True, exist_ok=True)
        (output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        return
    else:
        raise ValueError(args.command)
    if args.command in {"evaluate-events", "evaluate-ocr"}:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
