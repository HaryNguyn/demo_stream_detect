"""Validated inference configuration; relative paths are relative to the YAML."""
from dataclasses import dataclass, field
from pathlib import Path
import math

import yaml
from .roi import validate_roi


@dataclass
class PipelineConfig:
    vehicle_weights: str
    logo_weights: str = ""
    plate_weights: str = ""
    ocr_weights: str = ""
    ocr_config: str = ""
    ocr_engine: str = "cct"
    model_version: str = "baseline-001"
    vehicle_classes: dict[int, str] = field(default_factory=lambda: {
        2: "Ô tô con", 3: "Xe máy", 5: "Xe buýt", 7: "Xe tải",
    })
    detector_confidence: float = 0.2
    sample_interval_seconds: float = 0.5
    track_timeout_seconds: float = 2.0
    device: str = "cpu"
    roi: list[list[float]] = field(default_factory=list)
    detector_imgsz: int = 640
    detector_half: bool = False
    max_recognitions_per_frame: int = 2
    min_vehicle_width: int = 120
    min_plate_width: int = 32
    min_plate_sharpness: float = 0.0
    plate_confirm_reads: int = 3
    plate_confirm_score: float = 3.0
    plate_confirm_mean: float = 0.85
    crop_vehicle_roi: bool = True
    vehicle_roi_padding: float = 0.05
    plate_imgsz: int = 320
    plate_lower_fraction: float = 1.0
    plate_reuse_padding: float = 0.3
    plate_full_scan_interval: int = 5
    ocr_top_k: int = 4
    ocr_threads: int = 4
    ocr_batch_size: int = 8
    min_image_quality: float = 0.5
    min_evidence_detection: float = 0.6
    min_evidence_ocr: float = 0.85
    evidence_top_k: int = 5
    keep_low_confidence: bool = True
    plate_confirm_margin: float = 1.0
    plate_confirm_share: float = 0.0
    type_confirm_reads: int = 3
    type_confirm_mean: float = 0.7
    type_confirm_share: float = 0.7
    brand_confirm_reads: int = 3
    brand_confirm_mean: float = 0.7
    brand_confirm_share: float = 0.7
    quality_target_width: float = 100
    quality_target_height: float = 32
    quality_target_sharpness: float = 150
    quality_edge_margin: float = 2
    quality_expected_ratios: list[float] = field(default_factory=lambda: [1.4, 4.5])

    def __post_init__(self):
        self.device = str(self.device)
        self.roi = validate_roi(self.roi)
        self.vehicle_classes = {int(key): str(value) for key, value in self.vehicle_classes.items()}
        if not self.vehicle_classes or min(self.vehicle_classes) < 0:
            raise ValueError("vehicle_classes phải chứa ID lớp không âm.")
        for name in ("sample_interval_seconds", "track_timeout_seconds"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} phải > 0.")
        if not 0 < self.detector_confidence <= 1:
            raise ValueError("detector_confidence phải nằm trong (0, 1].")
        if self.ocr_engine not in {"cct", "crnn", "none"}:
            raise ValueError("ocr_engine phải là cct, crnn hoặc none.")
        for name in ("detector_imgsz", "max_recognitions_per_frame", "plate_confirm_reads",
                     "plate_imgsz", "plate_full_scan_interval", "ocr_top_k", "ocr_threads", "ocr_batch_size",
                     "type_confirm_reads", "brand_confirm_reads"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} phải là số nguyên >= 1.")
        for name in ("min_vehicle_width", "min_plate_width", "min_plate_sharpness"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} phải hữu hạn và >= 0.")
        if not math.isfinite(self.plate_confirm_score) or self.plate_confirm_score <= 0:
            raise ValueError("plate_confirm_score phải hữu hạn và > 0.")
        if not 0 < self.plate_confirm_mean <= 1:
            raise ValueError("plate_confirm_mean phải nằm trong (0, 1].")
        if not isinstance(self.detector_half, bool):
            raise ValueError("detector_half phải là true hoặc false.")
        if self.detector_half and self.device.lower() in {"cpu", "mps"}:
            raise ValueError("detector_half chỉ bật với thiết bị GPU hỗ trợ FP16.")
        for name in ("vehicle_roi_padding", "plate_reuse_padding"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} phải nằm trong [0, 1].")
        if not 0 < self.plate_lower_fraction <= 1:
            raise ValueError("plate_lower_fraction phải nằm trong (0, 1].")
        if not isinstance(self.crop_vehicle_roi, bool):
            raise ValueError("crop_vehicle_roi phải là true hoặc false.")
        for name in ("min_image_quality", "min_evidence_detection", "min_evidence_ocr",
                     "plate_confirm_share", "type_confirm_mean", "type_confirm_share",
                     "brand_confirm_mean", "brand_confirm_share"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} phải nằm trong [0, 1].")
        if isinstance(self.evidence_top_k, bool) or not isinstance(self.evidence_top_k, int) or not 3 <= self.evidence_top_k <= 5:
            raise ValueError("evidence_top_k phải là số nguyên từ 3 đến 5.")
        if not isinstance(self.keep_low_confidence, bool):
            raise ValueError("keep_low_confidence phải là true hoặc false.")
        for name in ("quality_target_width", "quality_target_height", "quality_target_sharpness",
                     "quality_edge_margin"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} phải hữu hạn và > 0.")
        if not math.isfinite(self.plate_confirm_margin) or self.plate_confirm_margin < 0:
            raise ValueError("plate_confirm_margin phải hữu hạn và >= 0.")
        if not self.quality_expected_ratios or any(
            not math.isfinite(value) or value <= 0 for value in self.quality_expected_ratios
        ):
            raise ValueError("quality_expected_ratios phải chứa các tỷ lệ hữu hạn > 0.")


def load_config(path):
    path = Path(path).resolve()
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Cấu hình phải là YAML mapping.")
    for name in ("vehicle_weights", "logo_weights", "plate_weights", "ocr_weights", "ocr_config"):
        if data.get(name):
            data[name] = str((path.parent / data[name]).resolve())
    return PipelineConfig(**data)
