"""Durable vehicle events for the local AI Box prototype."""
import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


class SQLiteEventStore:
    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with closing(sqlite3.connect(self.path)) as db:
            db.execute("""CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY, recorded_at TEXT NOT NULL,
                payload TEXT NOT NULL)""")
            db.commit()

    def append(self, row):
        event = {
            "event_id": str(uuid4()),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            **row,
        }
        event.pop("Ảnh xe", None)
        event.pop("Ảnh biển số", None)
        event["Xe ID"] = event.get("Biển số") or f"TRACK-{event.get('Track ID', 0)}"
        payload = {
            key: value for key, value in event.items()
            if key not in {"Danh sách ảnh xe", "Danh sách ảnh biển"}
        }
        payload["images"] = {"detected": [], "plate": []}
        with closing(sqlite3.connect(self.path)) as db:
            db.execute("INSERT INTO events VALUES (?, ?, ?)", (
                event["event_id"], event["recorded_at"],
                json.dumps(payload, ensure_ascii=False),
            ))
            db.commit()
        return event

    def update_images(self, event_id, saved):
        with closing(sqlite3.connect(self.path)) as db:
            with db:
                row = db.execute(
                    "SELECT payload FROM events WHERE event_id = ?", (event_id,)
                ).fetchone()
                if row is None:
                    raise KeyError(event_id)
                event = json.loads(row[0])
                event["images"] = saved
                db.execute(
                    "UPDATE events SET payload = ? WHERE event_id = ?",
                    (json.dumps(event, ensure_ascii=False), event_id),
                )

    def recent(self, limit=200):
        with closing(sqlite3.connect(self.path)) as db:
            rows = db.execute(
                "SELECT payload FROM events ORDER BY rowid DESC LIMIT ?", (limit,)
            ).fetchall()
        return [json.loads(row[0]) for row in rows]
