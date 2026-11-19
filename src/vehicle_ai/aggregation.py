"""Bounded multi-frame evidence and recognition aggregation."""
from dataclasses import dataclass, field

MAX_PLATE_SAMPLES = 30
MAX_PLATE_CANDIDATES = 20
TOP_SAMPLES = 3
MIN_SAMPLE_FRAME_DISTANCE = 5
MAX_IMAGE_SAMPLES = 30


def select_top_samples(samples):
    ranked = sorted(samples, key=lambda item: item["_score"], reverse=True)
    selected = []
    for candidate in ranked:
        if all(abs(candidate["_frame_idx"] - item["_frame_idx"]) >= MIN_SAMPLE_FRAME_DISTANCE
               for item in selected):
            selected.append(candidate)
        if len(selected) == TOP_SAMPLES:
            break
    selected_ids = {id(item) for item in selected}
    for candidate in ranked:
        if len(selected) == TOP_SAMPLES:
            break
        if id(candidate) not in selected_ids:
            selected.append(candidate)
            selected_ids.add(id(candidate))
    return sorted(selected, key=lambda item: item["_frame_idx"])


@dataclass
class Vehicle:
    time: float
    scores: dict[str, float] = field(default_factory=dict)
    brand: str = ""
    brand_score: float = 0.0
    plates: dict[str, list[float]] = field(default_factory=dict)
    detected_samples: list[dict] = field(default_factory=list)
    plate_samples: list[dict] = field(default_factory=list)
    last_seen: int = 0

    def plate_rank(self, plate):
        scores = self.plates[plate]
        return len(scores), max(scores)

    def add_plate(self, value, score):
        scores = self.plates.setdefault(value, [])
        scores.append(score)
        self.plates[value] = scores[-MAX_PLATE_SAMPLES:]
        if len(self.plates) > MAX_PLATE_CANDIDATES:
            del self.plates[min(self.plates, key=self.plate_rank)]

    def add_sample(self, kind, candidate):
        samples = self.detected_samples if kind == "detected" else self.plate_samples
        samples.append(candidate)
        if len(samples) > MAX_IMAGE_SAMPLES:
            samples.sort(key=lambda item: item["_score"], reverse=True)
            del samples[MAX_IMAGE_SAMPLES:]

    def row(self, camera, track_id):
        plate = max(self.plates, key=self.plate_rank) if self.plates else ""
        detected = select_top_samples(self.detected_samples)
        plates = select_top_samples(self.plate_samples)
        return {
            "Camera": camera, "Track ID": track_id,
            "Xe ID": plate or f"TRACK-{track_id}",
            "Xuất hiện (giây)": round(self.time, 3),
            "Loại xe": max(self.scores, key=self.scores.get) if self.scores else "",
            "Hãng": self.brand, "Điểm hãng": round(self.brand_score, 4),
            "Biển số": plate, "Điểm biển": round(max(self.plates[plate]), 4) if plate else 0.0,
            "Ảnh xe": detected[-1] if detected else None,
            "Ảnh biển số": plates[-1] if plates else None,
            "Danh sách ảnh xe": [{k: v for k, v in item.items() if not k.startswith("_")} for item in detected],
            "Danh sách ảnh biển": [{k: v for k, v in item.items() if not k.startswith("_")} for item in plates],
        }
