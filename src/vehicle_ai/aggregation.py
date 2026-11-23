"""Bounded multi-frame evidence and recognition aggregation."""
from dataclasses import dataclass, field
from .consensus import decide

MAX_PLATE_SAMPLES = 30
MAX_PLATE_CANDIDATES = 20
TOP_SAMPLES = 3
MIN_SAMPLE_FRAME_DISTANCE = 5
MAX_IMAGE_SAMPLES = 30


def select_top_samples(samples, limit=TOP_SAMPLES):
    ranked = sorted(samples, key=lambda item: item["_score"], reverse=True)
    selected = []
    for candidate in ranked:
        if all(abs(candidate["_frame_idx"] - item["_frame_idx"]) >= MIN_SAMPLE_FRAME_DISTANCE
               for item in selected):
            selected.append(candidate)
        if len(selected) == limit:
            break
    selected_ids = {id(item) for item in selected}
    for candidate in ranked:
        if len(selected) == limit:
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
    plate_box: tuple[float, float, float, float] | None = None
    plate_detection_score: float = 0.0
    plate_ocr_score: float = 0.0
    confirmed_plate: str = ""
    plate_searches: int = 0
    ocr_qualities: list[float] = field(default_factory=list)
    candidate_limit: int = MAX_IMAGE_SAMPLES
    brand_votes: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    type_votes: dict[str, dict] = field(default_factory=dict)
    classification_status: dict = field(default_factory=dict)
    plate_decision: dict = field(default_factory=dict)
    rejections: dict[str, int] = field(default_factory=dict)
    rejection_detail: dict[str, str] = field(default_factory=dict)

    def reject(self, reason, detail=""):
        self.rejections[reason] = self.rejections.get(reason, 0) + 1
        if detail:
            self.rejection_detail[reason] = detail

    def plate_rank(self, plate):
        scores = self.plates[plate]
        return sum(scores), len(scores), max(scores)

    def add_plate(self, value, score):
        if self.confirmed_plate:
            return
        scores = self.plates.setdefault(value, [])
        scores.append(score)
        self.plates[value] = scores[-MAX_PLATE_SAMPLES:]
        if len(self.plates) > MAX_PLATE_CANDIDATES:
            del self.plates[min(self.plates, key=self.plate_rank)]

    def confirm_plate(self, min_reads, min_score, min_mean, min_margin=1.0, min_share=0.0,
                      image_ready=True):
        if self.confirmed_plate or not self.plates:
            return
        self.plate_decision = decide(self.vote_summaries()["plate"], min_reads=min_reads,
                                     min_score=min_score, min_mean=min_mean,
                                     min_margin=min_margin, min_share=min_share)
        if not image_ready:
            self.plate_decision["confirmed"] = False
            self.plate_decision["unmet"].append("image_quality")
        if self.plate_decision["confirmed"]:
            self.confirmed_plate = self.plate_decision["best"]

    def add_type(self, name, confidence, weight):
        values = self.type_votes.setdefault(name, {"score": 0, "count": 0, "confidence_sum": 0})
        values["score"] += confidence * weight
        values["count"] += 1
        values["confidence_sum"] += confidence
        self.scores[name] = values["score"]

    def vote_summaries(self):
        return {
            "plate": {name: {"score": sum(values), "count": len(values), "confidence_sum": sum(values)}
                      for name, values in self.plates.items()},
            "vehicle_type": {name: dict(values) for name, values in self.type_votes.items()},
            "brand": {name: {"score": sum(v[1] for v in values), "count": len(values),
                             "confidence_sum": sum(v[0] for v in values)}
                      for name, values in self.brand_votes.items()},
        }

    def add_brand(self, name, confidence, quality):
        votes = self.brand_votes.setdefault(name, [])
        votes.append((confidence, confidence * quality))
        self.brand_votes[name] = votes[-MAX_PLATE_SAMPLES:]
        if len(self.brand_votes) > MAX_PLATE_CANDIDATES:
            del self.brand_votes[min(self.brand_votes, key=lambda key: sum(v[1] for v in self.brand_votes[key]))]
        self.brand = max(self.brand_votes, key=lambda key: sum(v[1] for v in self.brand_votes[key]))
        winner = self.brand_votes[self.brand]
        self.brand_score = sum(v[0] for v in winner) / len(winner)

    def keeps_sample(self, kind, score):
        """True when an encoded sample could still enter the top-k evidence list."""
        samples = self.detected_samples if kind == "detected" else self.plate_samples
        return len(samples) < self.candidate_limit or score > min(
            item["_score"] for item in samples)

    def add_sample(self, kind, candidate):
        samples = self.detected_samples if kind == "detected" else self.plate_samples
        samples[:] = [sample for sample in samples if sample["_frame_idx"] != candidate["_frame_idx"]]
        samples.append(candidate)
        if len(samples) > self.candidate_limit:
            samples.sort(key=lambda item: item["_score"], reverse=True)
            del samples[self.candidate_limit:]

    def row(self, camera, track_id, *, min_image_quality=None, min_detection=0.0, min_ocr=0.0,
            top_k=TOP_SAMPLES, keep_low_confidence=False):
        plate = max(self.plates, key=self.plate_rank) if self.plates else ""
        detected = select_top_samples(self.detected_samples)
        plates = select_top_samples(self.plate_samples)
        if min_image_quality is not None:
            def qualified(samples, is_plate=False):
                candidates = [
                    item for item in samples
                    if item.get("quality_score", 0) >= min_image_quality
                    and item.get("detection_score", 0) >= min_detection
                    and (not is_plate or (
                        item.get("recognized_plate") == plate
                        and item.get("ocr_score", 0) >= min_ocr
                    ))
                ]
                return select_top_samples(candidates, top_k)

            detected = qualified(self.detected_samples)
            plates = qualified(self.plate_samples, is_plate=True)
        qualified_plate = bool(plates)
        if keep_low_confidence:
            if not detected:
                detected = select_top_samples(self.detected_samples, top_k)
            if not plates:
                plates = select_top_samples(self.plate_samples, top_k)
        best_detected = max(detected, key=lambda item: item["_score"], default=None)
        best_plate = max(plates, key=lambda item: item["_score"], default=None)
        reasons, details = dict(self.rejections), dict(self.rejection_detail)
        if self.plates and not qualified_plate:
            reasons["quality_below_threshold"] = reasons.get("quality_below_threshold", 0) + 1
        if self.plates and not self.confirmed_plate:
            reasons.setdefault("plate_not_confirmed", 1)
            details.setdefault("plate_not_confirmed", ",".join(self.plate_decision.get("unmet", [])))
        if not self.plates and not reasons:
            reasons["no_plate_read"] = 1
        return {
            "Camera": camera, "Track ID": track_id,
            "Xe ID": plate or f"TRACK-{track_id}",
            "Xuất hiện (giây)": round(self.time, 3),
            "Loại xe": max(self.scores, key=self.scores.get) if self.scores else "",
            "Hãng": self.brand, "Điểm hãng": round(self.brand_score, 4),
            "Biển số": plate, "Điểm biển": round(max(self.plates[plate]), 4) if plate else 0.0,
            "Ảnh xe": best_detected,
            "Ảnh biển số": best_plate,
            "confidence_status": "confirmed" if self.confirmed_plate and qualified_plate else "low",
            "plate_votes": {text: round(sum(values), 4) for text, values in self.plates.items()},
            "brand_votes": {name: round(sum(v[1] for v in votes), 4) for name, votes in self.brand_votes.items()},
            "vote_summaries": self.vote_summaries(),
            "classification_status": self.classification_status,
            "plate_decision": self.plate_decision,
            "source_track_ids": [track_id],
            "rejection_reasons": reasons,
            "rejection_detail": details,
            "Danh sách ảnh xe": [{k: v for k, v in item.items() if not k.startswith("_")} for item in detected],
            "Danh sách ảnh biển": [{k: v for k, v in item.items() if not k.startswith("_")} for item in plates],
        }
