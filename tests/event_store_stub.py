"""In-memory test double, never used by application runtime."""
from copy import deepcopy
from datetime import datetime, timezone
from uuid import uuid4


class EventStoreStub:
    def __init__(self):
        self.rows = []

    def append(self, row):
        event = {
            **deepcopy(row), "event_id": str(uuid4()),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "Xe ID": row.get("Biển số") or f"TRACK-{row.get('Track ID', 0)}",
        }
        stored = {key: value for key, value in event.items()
                  if key not in {"Ảnh xe", "Ảnh biển số", "Danh sách ảnh xe", "Danh sách ảnh biển"}}
        stored["images"] = {"detected": [], "plate": []}
        self.rows.insert(0, stored)
        return event

    def update_images(self, event_id, images):
        row = next(row for row in self.rows if row["event_id"] == event_id)
        row["images"] = deepcopy(images)

    def recent(self, limit=200):
        return deepcopy(self.rows[:limit])
