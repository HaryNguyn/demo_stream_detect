"""File-based experiment output, deliberately isolated from production databases."""
import base64
import json
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4

from .domain import event_from_row
from .evidence import IMAGE_FIELDS


class ExperimentWriter:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=False)
        self.stream = (self.directory / "events.jsonl").open("x", encoding="utf-8")
        self.count = 0

    def append(self, row):
        event = dict(row["event"])
        event["evidence"] = []
        event_dir = self.directory / "evidence" / event["event_id"]
        for kind, field in (("vehicle", "Danh sách ảnh xe"), ("plate", "Danh sách ảnh biển")):
            for index, image in enumerate(row.get(field, []), 1):
                event_dir.mkdir(parents=True, exist_ok=True)
                path = event_dir / f"{kind}_{index:02d}.jpg"
                path.write_bytes(base64.b64decode(image["data"], validate=True))
                event["evidence"].append({
                    "kind": kind, "path": path.relative_to(self.directory).as_posix(),
                    "quality_score": image.get("quality_score"),
                    "frame_idx": image.get("frame_idx"),
                })
        self.stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        self.stream.flush()
        self.count += 1

    def close(self):
        self.stream.close()


class MergedExperimentStore:
    """File-only repository used by the same EvidenceRecorder as the web."""
    def __init__(self, directory, source_id, model_version):
        self.directory = Path(directory)
        self.source_id, self.model_version = source_id, model_version
        self.rows = {}

    def append(self, row):
        event = {**row, "event_id": str(uuid4()), "recorded_at": datetime.now(timezone.utc).isoformat()}
        self.rows[event["event_id"]] = {
            key: value for key, value in event.items()
            if key not in {*IMAGE_FIELDS.values(), "Ảnh xe", "Ảnh biển số", "event"}
        }
        return event

    def update_images(self, event_id, images):
        self.rows[event_id]["images"] = images
        self.rows[event_id].update(images.get("_recognition", {}))

    def close(self):
        with (self.directory / "events.merged.jsonl").open("x", encoding="utf-8") as stream:
            for row in self.rows.values():
                event = event_from_row(
                    row, source_id=self.source_id, model_version=self.model_version,
                    reason=row.get("finalization_reason", "end_of_source"),
                )
                event.update(event_id=row["event_id"], recorded_at=row["recorded_at"], event_stage="merged",
                             raw_event_ids=row.get("raw_event_ids", []))
                event["evidence"] = [
                    {"kind": "vehicle" if kind == "detected" else kind,
                     **{key: value for key, value in item.items() if key != "path"},
                     "path": Path(item["path"]).relative_to(self.directory).as_posix()}
                    for kind in IMAGE_FIELDS for item in row.get("images", {}).get(kind, [])
                ]
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")
