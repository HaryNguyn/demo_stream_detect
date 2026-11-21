"""Stable JSON event schema, independent of UI column names."""
from datetime import datetime, timezone
from uuid import uuid4


def event_from_row(row, *, source_id, model_version, reason):
    return {
        "schema_version": "1.0",
        "event_stage": "raw",
        "event_id": str(uuid4()),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "source_id": source_id,
        "camera_id": row["Camera"],
        "track_id": row["Track ID"],
        "started_at_seconds": row["Xuất hiện (giây)"],
        "ended_at_seconds": row["Kết thúc (giây)"],
        "vehicle_type": row["Loại xe"],
        "brand": row["Hãng"] or None,
        "brand_confidence": row["Điểm hãng"],
        "plate": row["Biển số"] or None,
        "plate_confidence": row["Điểm biển"],
        "status": row["status"],
        "confidence_status": row.get("confidence_status", "unknown"),
        "plate_decision": row.get("plate_decision", {}),
        "classification_status": row.get("classification_status", {}),
        "vote_summaries": row.get("vote_summaries", {}),
        "source_track_ids": row.get("source_track_ids", [row["Track ID"]]),
        "rejection_reasons": row.get("rejection_reasons", {}),
        "rejection_detail": row.get("rejection_detail", {}),
        "finalization_reason": reason,
        "model_version": model_version,
        "evidence": [],
    }
