"""File-based experiment output, deliberately isolated from production databases."""
import base64
import json
from pathlib import Path


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
