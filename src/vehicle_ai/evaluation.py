"""Event metrics: one-to-one temporal matching, independent of predicted plate."""
import json
import math
from pathlib import Path
import unicodedata

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


def normalize_label(text):
    return " ".join(unicodedata.normalize("NFKC", text or "").casefold().split())


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
        tags = row.get("tags")
        if tags is not None and (
            not isinstance(tags, list)
            or any(not isinstance(tag, str) or not tag.strip() for tag in tags)
        ):
            raise ValueError("tags phải là danh sách chuỗi không rỗng.")
        if "notes" in row and not isinstance(row["notes"], str):
            raise ValueError("notes phải là chuỗi.")
    return rows


def temporal_iou(a, b):
    start = max(a["started_at_seconds"], b["started_at_seconds"])
    end = min(a["ended_at_seconds"], b["ended_at_seconds"])
    union = max(a["ended_at_seconds"], b["ended_at_seconds"]) - min(
        a["started_at_seconds"], b["started_at_seconds"])
    if union == 0:
        return 1.0
    return max(0.0, end - start) / union


def evaluate_events(truth, predicted, iou_threshold=0.3, *, known_brands=None):
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
    taxonomy = {normalize_label(name) for name in known_brands} if known_brands is not None else None
    brand_groups = {
        name: {"ground_truth_events": 0, "matched_events": 0, "correct": 0,
               "confident_wrong": 0, "predicted_known_brand": 0}
        for name in ("known", "unknown", "taxonomy_unspecified")
    }

    def brand_group(row):
        if taxonomy is None:
            return "taxonomy_unspecified"
        return "known" if normalize_label(row.get("brand")) in taxonomy else "unknown"

    for row in truth:
        if normalize_label(row.get("brand")):
            brand_groups[brand_group(row)]["ground_truth_events"] += 1

    def tag_names(row):
        return {normalize_label(tag) for tag in row.get("tags") or [] if normalize_label(tag)}

    tag_groups = {}
    for row in truth:
        for tag in tag_names(row):
            group = tag_groups.setdefault(tag, {
                "ground_truth_events": 0, "matched_events": 0, "plate_readable_ground_truth": 0,
                "plate_matched_readable": 0, "plate_exact_matches": 0,
            })
            group["ground_truth_events"] += 1
            group["plate_readable_ground_truth"] += int(bool(normalize_plate(row.get("plate"))))

    def confidence_group(row):
        status = row.get("confidence_status", "unknown")
        return status if status in {"confirmed", "low"} else "unknown"

    groups = {}
    for name in ("confirmed", "low", "unknown"):
        indices = [j for j, row in enumerate(predicted) if confidence_group(row) == name]
        groups[name] = {
            "predicted_events": len(indices), "share_of_predictions": len(indices) / m if m else 0,
            "matched_events": 0, "plate_readable_matched": 0, "plate_exact_matches": 0,
            "extra_events": sum(j not in matched_pred for j in indices),
            "possible_duplicate_events": sum(
                j not in matched_pred and bool((weights[:, j] > 0).any()) for j in indices),
            "_edits": 0, "_chars": 0,
        }
    matches = []
    for i, j in pairs:
        gt, pred = truth[i], predicted[j]
        target, text = normalize_plate(gt.get("plate")), normalize_plate(pred.get("plate"))
        group = groups[confidence_group(pred)]
        group["matched_events"] += 1
        if target:
            readable_matched += 1
            exact += target == text
            edits += edit_distance(target, text)
            chars += len(target)
            group["plate_readable_matched"] += 1
            group["plate_exact_matches"] += int(target == text)
            group["_edits"] += edit_distance(target, text)
            group["_chars"] += len(target)
        for field, count in field_counts.items():
            if normalize_label(gt.get(field)):
                count[1] += 1
                count[0] += normalize_label(gt[field]) == normalize_label(pred.get(field))
        if normalize_label(gt.get("brand")):
            brand = brand_groups[brand_group(gt)]
            brand["matched_events"] += 1
            correct = normalize_label(gt["brand"]) == normalize_label(pred.get("brand"))
            brand["correct"] += int(correct)
            brand["confident_wrong"] += int(
                not correct and pred.get("classification_status", {}).get("brand", {}).get("confirmed", False))
            brand["predicted_known_brand"] += int(taxonomy is not None and normalize_label(pred.get("brand")) in taxonomy)
        for tag in tag_names(gt):
            group = tag_groups[tag]
            group["matched_events"] += 1
            if target:
                group["plate_matched_readable"] += 1
                group["plate_exact_matches"] += int(target == text)
        matches.append({"truth_id": gt["event_id"], "prediction_id": pred["event_id"],
                        "temporal_iou": float(overlap[i, j]), "confidence_group": confidence_group(pred)})
    for group in groups.values():
        denominator = group["plate_readable_matched"]
        group["plate_exact_match_on_matched"] = group["plate_exact_matches"] / denominator if denominator else None
        group["plate_correct_over_total"] = f"{group['plate_exact_matches']}/{denominator}" if denominator else None
        group["plate_cer_on_matched"] = group["_edits"] / group["_chars"] if group["_chars"] else None
        del group["_edits"], group["_chars"]
    classified = groups["confirmed"]["predicted_events"] + groups["low"]["predicted_events"]
    for group in brand_groups.values():
        total = group["matched_events"]
        group["accuracy_on_matched"] = group["correct"] / total if total else None
        group["correct_over_total"] = f"{group['correct']}/{total}" if total else None
        group["missed_events"] = group["ground_truth_events"] - total
    for group in tag_groups.values():
        total = group["ground_truth_events"]
        readable = group["plate_readable_ground_truth"]
        group["missed_events"] = total - group["matched_events"]
        group["event_recall"] = group["matched_events"] / total if total else None
        group["plate_exact_match_on_matched"] = (
            group["plate_exact_matches"] / group["plate_matched_readable"]
            if group["plate_matched_readable"] else None)
        group["plate_correct_over_total"] = (
            f"{group['plate_exact_matches']}/{readable}" if readable else None)
        group["plate_end_to_end_recall"] = group["plate_exact_matches"] / readable if readable else None
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
        "plate_readable_matched": readable_matched,
        "plate_correct_over_total": f"{exact}/{readable_matched}" if readable_matched else None,
        "plate_exact_match_on_matched": exact / readable_matched if readable_matched else None,
        "plate_end_to_end_recall": exact / readable_gt if readable_gt else None,
        "plate_cer_on_matched": edits / chars if chars else None,
        "confirmation_rate": groups["confirmed"]["predicted_events"] / classified if classified else None,
        "confidence_groups": groups,
        "field_counts": {field: {"correct": c, "total": total,
                                 "correct_over_total": f"{c}/{total}" if total else None}
                         for field, (c, total) in field_counts.items()},
        "brand_groups": brand_groups,
        "tag_groups": tag_groups,
        "brand_taxonomy": sorted(taxonomy) if taxonomy is not None else None,
        "prediction_stages": sorted({row.get("event_stage", "unspecified") for row in predicted}),
        **{f"{field}_accuracy_on_matched": c / total if total else None
           for field, (c, total) in field_counts.items()},
        "matching": {"method": "one_to_one_temporal_iou", "threshold": iou_threshold},
        "matches": matches,
        "unmatched_truth_ids": [row["event_id"] for i, row in enumerate(truth) if i not in {p[0] for p in pairs}],
        "unmatched_prediction_ids": [row["event_id"] for j, row in enumerate(predicted) if j not in matched_pred],
    }


def compare_reports(before, after):
    fields = ("matched_events", "missed_events", "extra_events", "possible_duplicate_events",
              "event_precision", "event_recall", "plate_exact_matches", "plate_exact_match_on_matched",
              "plate_cer_on_matched", "confirmation_rate")
    delta = {field: after[field] - before[field]
             if after[field] is not None and before[field] is not None else None for field in fields}
    delta["confidence_groups"] = {
        group: {
            field: after["confidence_groups"][group][field] - before["confidence_groups"][group][field]
            if after["confidence_groups"][group][field] is not None
            and before["confidence_groups"][group][field] is not None else None
            for field in ("predicted_events", "plate_exact_matches", "plate_exact_match_on_matched",
                          "plate_cer_on_matched")
        }
        for group in ("confirmed", "low", "unknown")
    }
    delta["brand_groups"] = {
        name: {field: after["brand_groups"][name][field] - before["brand_groups"][name][field]
               if after["brand_groups"][name][field] is not None
               and before["brand_groups"][name][field] is not None else None
               for field in ("correct", "matched_events", "accuracy_on_matched", "confident_wrong")}
        for name in ("known", "unknown", "taxonomy_unspecified")
    }
    tag_fields = ("ground_truth_events", "matched_events", "missed_events", "event_recall",
                  "plate_exact_matches", "plate_end_to_end_recall")
    delta["tag_groups"] = {
        tag: {field: after["tag_groups"][tag][field] - before["tag_groups"][tag][field]
              if tag in after["tag_groups"] and tag in before["tag_groups"]
              and after["tag_groups"][tag][field] is not None
              and before["tag_groups"][tag][field] is not None else None
              for field in tag_fields}
        for tag in sorted(set(before["tag_groups"]) | set(after["tag_groups"]))
    }
    return delta
