"""Event metrics: one-to-one temporal matching, independent of predicted plate."""
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, char in enumerate(a, 1):
        current = [i]
        for j, other in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (char != other)))
        previous = current
    return previous[-1]


def normalize_plate(text):
    return "".join(char for char in (text or "").upper() if char.isascii() and char.isalnum())


def read_events(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    ids = set()
    for row in rows:
        for key in ("event_id", "source_id", "camera_id", "started_at_seconds", "ended_at_seconds"):
            if key not in row:
                raise ValueError(f"Sự kiện thiếu {key}.")
        if row["event_id"] in ids:
            raise ValueError("event_id trùng trong file.")
        ids.add(row["event_id"])
        start, end = row["started_at_seconds"], row["ended_at_seconds"]
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start <= end):
            raise ValueError("Thời gian sự kiện không hợp lệ.")
    return rows


def temporal_iou(a, b):
    start = max(a["started_at_seconds"], b["started_at_seconds"])
    end = min(a["ended_at_seconds"], b["ended_at_seconds"])
    union = max(a["ended_at_seconds"], b["ended_at_seconds"]) - min(
        a["started_at_seconds"], b["started_at_seconds"])
    if union == 0:
        return 1.0
    return max(0.0, end - start) / union


def evaluate_events(truth, predicted, iou_threshold=0.3):
    if not truth:
        raise ValueError("Ground truth rỗng; không thể báo cáo accuracy.")
    if not 0 < iou_threshold <= 1:
        raise ValueError("IoU threshold phải nằm trong (0, 1].")
    n, m = len(truth), len(predicted)
    weights = np.zeros((n, m))
    overlap = np.zeros((n, m))
    for i, gt in enumerate(truth):
        for j, pred in enumerate(predicted):
            if (gt["source_id"], gt["camera_id"]) != (pred["source_id"], pred["camera_id"]):
                continue
            score = temporal_iou(gt, pred)
            overlap[i, j] = score
            if score >= iou_threshold:
                weights[i, j] = min(n, m) + 1 + score
    pairs = []
    if m:
        ri, ci = linear_sum_assignment(-weights)
        pairs = [(int(i), int(j)) for i, j in zip(ri, ci) if weights[i, j] > 0]
    matched_pred = {j for _, j in pairs}
    tp = len(pairs)
    precision, recall = tp / m if m else 0.0, tp / n
    readable_gt = sum(bool(normalize_plate(row.get("plate"))) for row in truth)
    exact = readable_matched = edits = chars = 0
    field_counts = {field: [0, 0] for field in ("vehicle_type", "brand")}
    matches = []
    for i, j in pairs:
        gt, pred = truth[i], predicted[j]
        target, text = normalize_plate(gt.get("plate")), normalize_plate(pred.get("plate"))
        if target:
            readable_matched += 1
            exact += target == text
            edits += edit_distance(target, text)
            chars += len(target)
        for field, count in field_counts.items():
            if gt.get(field):
                count[1] += 1
                count[0] += gt[field] == pred.get(field)
        matches.append({"truth_id": gt["event_id"], "prediction_id": pred["event_id"],
                        "temporal_iou": float(overlap[i, j])})
    return {
        "ground_truth_events": n, "predicted_events": m, "matched_events": tp,
        "missed_events": n - tp, "extra_events": m - tp,
        "possible_duplicate_events": sum(
            j not in matched_pred and bool((weights[:, j] > 0).any()) for j in range(m)
        ),
        "event_precision": precision, "event_recall": recall,
        "event_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "plate_readable_ground_truth": readable_gt,
        "plate_exact_matches": exact,
        "plate_exact_match_on_matched": exact / readable_matched if readable_matched else None,
        "plate_end_to_end_recall": exact / readable_gt if readable_gt else None,
        "plate_cer_on_matched": edits / chars if chars else None,
        **{f"{field}_accuracy_on_matched": c / total if total else None
           for field, (c, total) in field_counts.items()},
        "matching": {"method": "one_to_one_temporal_iou", "threshold": iou_threshold},
        "matches": matches,
        "unmatched_truth_ids": [row["event_id"] for i, row in enumerate(truth) if i not in {p[0] for p in pairs}],
        "unmatched_prediction_ids": [row["event_id"] for j, row in enumerate(predicted) if j not in matched_pred],
    }
