"""Top evidence, confidence labels and source-scoped duplicate suppression."""
from base64 import b64decode
from collections import Counter
from hashlib import sha256
import re

from .evidence import IMAGE_FIELDS, save_event_images
from .consensus import classification_decisions, decide, merge_votes
from .config import PipelineConfig


def identity_key(row):
    plate = re.sub(r"[^A-Z0-9]", "", str(row.get("Biển số") or "").upper())
    return str(row.get("Camera") or ""), plate


def image_score(row, kind="plate"):
    return max((image.get("quality_score", 0) for image in (row.get("images") or {}).get(kind, [])),
               default=0)


def unique_results(rows):
    selected = {}
    for original in rows:
        row = {**original, **(original.get("images") or {}).get("_recognition", {})}
        key = identity_key(row)
        if not key[1]:
            if row.get("confidence_status") != "low":
                continue
            key = (key[0], f"unread:{row['event_id']}")
        rank = (row.get("confidence_status") != "low", image_score(row))
        if key not in selected or rank > (
            selected[key].get("confidence_status") != "low", image_score(selected[key]),
        ):
            selected[key] = row
    return list(selected.values())


def sample_id(sample):
    if sample.get("image_id"):
        return sample["image_id"]
    return sha256(b64decode(sample["data"], validate=True)).hexdigest()


def best_images(candidates, limit):
    unique = {}
    for sample in candidates:
        key = sample_id(sample)
        if key not in unique or sample.get("quality_score", 0) > unique[key].get("quality_score", 0):
            unique[key] = sample
    return sorted(unique.values(), key=lambda item: item.get("quality_score", 0), reverse=True)[:limit]


class EvidenceRecorder:
    def __init__(self, store, root, *, min_quality=.5, min_detection=.6, min_ocr=.85,
                 top_k=5, keep_low_confidence=False, config=None):
        self.store, self.root = store, root
        self.min_quality, self.min_detection, self.min_ocr = min_quality, min_detection, min_ocr
        self.top_k, self.keep_low_confidence = top_k, keep_low_confidence
        self.recorded = {}
        self.config = config or PipelineConfig(vehicle_weights="unused")
        self.stats = Counter()

    def _merge_decisions(self, row, previous):
        row = dict(row)
        incoming = row.get("vote_summaries", {})
        existing = (previous or {}).get("vote_summaries", {})
        old_ids = set((previous or {}).get("source_track_ids", []))
        new_ids = set(row.get("source_track_ids", [row.get("Track ID")])) - {None}
        reasons = dict(row.get("rejection_reasons") or {})
        details = dict(row.get("rejection_detail") or {})
        same_track = bool(old_ids & new_ids)
        for name, count in ((previous or {}).get("rejection_reasons") or {}).items():
            reasons[name] = max(reasons.get(name, 0), count) if same_track else reasons.get(name, 0) + count
        for name, value in ((previous or {}).get("rejection_detail") or {}).items():
            details.setdefault(name, value)
        if previous and old_ids and new_ids and not (old_ids & new_ids):
            reasons["merged_duplicate"] = reasons.get("merged_duplicate", 0) + 1
            details["merged_duplicate"] = f"gộp vào {previous.get('event_id', '')}"
            self.stats["merged_duplicate"] += 1
        if old_ids & new_ids:
            summaries = existing
        else:
            summaries = {name: merge_votes(existing.get(name, {}), incoming.get(name, {}))
                         for name in ("plate", "vehicle_type", "brand")}
        row["source_track_ids"] = sorted(old_ids | new_ids)
        row["raw_event_ids"] = sorted(set((previous or {}).get("raw_event_ids", [])) | set(row.get("raw_event_ids", [])))
        row["vote_summaries"] = summaries
        row["plate_votes"] = {name: round(values["score"], 4) for name, values in summaries.get("plate", {}).items()}
        row["brand_votes"] = {name: round(values["score"], 4) for name, values in summaries.get("brand", {}).items()}
        row["classification_status"] = classification_decisions(summaries, self.config)
        for name, field in (("vehicle_type", "Loại xe"), ("brand", "Hãng")):
            decision = row["classification_status"][name]
            if decision["best"]:
                row[field] = decision["best"]
                if name == "brand":
                    row["Điểm hãng"] = decision["mean"]
        if summaries.get("plate"):
            config = self.config
            decision = decide(summaries["plate"], min_reads=config.plate_confirm_reads,
                              min_score=config.plate_confirm_score, min_mean=config.plate_confirm_mean,
                              min_margin=config.plate_confirm_margin, min_share=config.plate_confirm_share)
            photos = [*row.get("Danh sách ảnh biển", []), *(previous or {}).get("images", {}).get("plate", [])]
            ready = any(
                identity_key({"Biển số": sample.get("recognized_plate")})[1] == identity_key(row)[1]
                and sample.get("quality_score", 0) >= self.min_quality
                and sample.get("detection_score", 0) >= self.min_detection
                and sample.get("ocr_score", 0) >= self.min_ocr for sample in photos
            )
            if not ready:
                decision["confirmed"] = False
                decision["unmet"].append("image_quality")
                reasons["quality_below_threshold"] = reasons.get("quality_below_threshold", 0) + 1
                details["quality_below_threshold"] = ",".join(decision["unmet"])
                self.stats["quality_below_threshold"] += 1
            row["plate_decision"] = decision
            row["confidence_status"] = "confirmed" if decision["confirmed"] else "low"
        if previous and previous.get("confidence_status") == "confirmed":
            row["confidence_status"] = "confirmed"
            row["Điểm biển"] = max(row.get("Điểm biển", 0), previous.get("Điểm biển", 0))
        row["rejection_reasons"] = reasons
        row["rejection_detail"] = details
        return row

    def record(self, row):
        key = identity_key(row)
        lookup_key = key if key[1] else (key[0], f"track:{row.get('Track ID')}")
        previous = self.recorded.get(lookup_key)
        row = self._merge_decisions(row, previous)
        fallback = self.keep_low_confidence and row.get("confidence_status") == "low"
        if row.get("confidence_status") == "low" and not self.keep_low_confidence:
            self.stats["low_dropped"] += 1
            return
        if not key[1]:
            if not fallback or "Track ID" not in row:
                return
            key = (key[0], f"track:{row['Track ID']}")
        selected = {}
        for kind, field in IMAGE_FIELDS.items():
            candidates = [sample for sample in row.get(field, []) or []
                          if sample.get("encoding") == "base64" and sample.get("data")]
            qualified = [
                sample for sample in candidates
                if sample.get("quality_score", 0) >= self.min_quality
                and sample.get("detection_score", 0) >= self.min_detection
                and (kind != "plate" or (
                    identity_key({"Biển số": sample.get("recognized_plate")})[1] == identity_key(row)[1]
                    and sample.get("ocr_score", 0) >= self.min_ocr
                ))
            ]
            selected[kind] = best_images(candidates if fallback else qualified, self.top_k)
        if not selected["plate"] and not fallback and previous is None:
            self.stats["no_qualified_evidence"] += 1
            return
        payload = {**row, "confidence_status": "low" if fallback else "confirmed"}
        for kind, field in IMAGE_FIELDS.items():
            prior = (previous or {}).get("images", {}).get(kind, [])
            if previous and previous.get("confidence_status") == "low" and not fallback:
                prior = []  # Replace provisional evidence when the identity becomes confirmed.
            payload[field] = best_images([*prior, *selected[kind]], self.top_k)
        if previous is None:
            event = self.store.append(payload)
        else:
            event = {**payload, "event_id": previous["event_id"], "recorded_at": previous["recorded_at"],
                     "Xe ID": previous.get("Xe ID")}
            event["Xuất hiện (giây)"] = min(row.get("Xuất hiện (giây)", 0), previous.get("Xuất hiện (giây)", 0))
            event["Kết thúc (giây)"] = max(row.get("Kết thúc (giây)", 0), previous.get("Kết thúc (giây)", 0))
        _, _, saved = save_event_images(event, self.root, previous_images=(previous or {}).get("images"))
        self.store.update_images(event["event_id"], saved)
        self.recorded[key] = {
            **{name: value for name, value in event.items()
               if name not in {*IMAGE_FIELDS.values(), "Ảnh xe", "Ảnh biển số", "event"}},
            "images": saved,
        }
