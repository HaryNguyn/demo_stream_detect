"""Model settings for browser-based inference."""
import json
from pathlib import Path

from .config import PipelineConfig, apply_speed_preset, load_config
from .roi import validate_roi


def prepare_models(options, *, ocr_only=False):
    from .inference import optional_yolo
    from .model_loader import load_recognizer
    from .ocr import VietnamesePlateRecognizer

    use_config = options.get("mode", "yaml") == "yaml"
    if use_config:
        config = load_config(options.get("config") or "configs/vietnam-ocr.yaml")
    else:
        config = PipelineConfig(
            vehicle_weights=options.get("vehicle_weights") or "models/yolo11n.pt",
            logo_weights=options.get("logo_weights", ""),
            plate_weights=options.get("plate_weights", ""),
            detector_imgsz=int(options.get("detector_imgsz") or 640),
            max_recognitions_per_frame=int(options.get("max_recognitions_per_frame") or 2),
            min_vehicle_width=int(options.get("min_vehicle_width") or 120),
            plate_confirm_score=float(options.get("plate_confirm_score") or 3.0),
            plate_imgsz=int(options.get("plate_imgsz") or 320),
            plate_lower_fraction=float(options.get("plate_lower_fraction") or 1.0),
            ocr_threads=int(options.get("ocr_threads") or 4),
            ocr_batch_size=int(options.get("ocr_batch_size") or 8),
            min_image_quality=float(options.get("min_image_quality") or .5),
            min_evidence_detection=float(options.get("min_evidence_detection") or .6),
            min_evidence_ocr=float(options.get("min_evidence_ocr") or .85),
            evidence_top_k=int(options.get("evidence_top_k") or 5),
            plate_confirm_margin=float(options.get("plate_confirm_margin") or 1.0),
            keep_low_confidence=options.get("keep_low_confidence", "true") == "true",
        )
    if options.get("roi_path"):
        roi = json.loads(Path(options["roi_path"]).read_text(encoding="utf-8"))
        config.roi = validate_roi(roi["points"])
    apply_speed_preset(config, options.get("speed_preset"))
    logo = None if ocr_only else optional_yolo(config.logo_weights)
    plate = None if ocr_only else optional_yolo(config.plate_weights)
    recognizer = None
    if use_config:
        recognizer = load_recognizer(config)
    elif ocr_only or plate is not None:
        weights, vocabulary = options.get("ocr_weights"), options.get("ocr_config")
        if not weights or not vocabulary:
            raise ValueError("Nhập đủ model OCR ONNX và cấu hình YAML.")
        recognizer = VietnamesePlateRecognizer(weights, vocabulary,
                                              threads=config.ocr_threads, batch_size=config.ocr_batch_size)
    return config, logo, plate, recognizer
