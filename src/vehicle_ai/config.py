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


def load_config(path):
    path = Path(path).resolve()
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Cấu hình phải là YAML mapping.")
    for name in ("vehicle_weights", "logo_weights", "plate_weights", "ocr_weights", "ocr_config"):
        if data.get(name):
            data[name] = str((path.parent / data[name]).resolve())
    return PipelineConfig(**data)
