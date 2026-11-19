"""Durable vehicle events stored in MariaDB."""
import os
import sys
import json
from datetime import datetime, timezone
from uuid import uuid4

import pymysql


def get_setting(name, default=None, required=False):
    value = os.environ.get(name)
    if value:
        return value
    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Environment",
            ) as key:
                value, _ = winreg.QueryValueEx(key, name)
                if value:
                    return value
        except (OSError, ValueError):
            pass
    if required:
        raise RuntimeError(
            f"Thiếu cấu hình MariaDB: {name}. "
            "Đặt biến môi trường hoặc khởi động lại VS Code."
        )
    return default


class EventStore:
    def __init__(self):
        self.config = {
            "host": get_setting("MARIADB_HOST", "127.0.0.1"),
            "port": int(get_setting("MARIADB_PORT", "3306")),
            "database": get_setting("MARIADB_DATABASE", "lane_demo"),
            "user": get_setting("MARIADB_USER", required=True),
            "password": get_setting("MARIADB_PASSWORD", required=True),
            "charset": "utf8mb4",
            "autocommit": True,
        }

    def connect(self):
        return pymysql.connect(**self.config)

    def append(self, row):
        now = datetime.now(timezone.utc)
        event = {
            "event_id": str(uuid4()),
            "recorded_at": now.isoformat(),
            **row,
        }
        event.pop("Ảnh xe", None)
        event.pop("Ảnh biển số", None)
        plate = event.get("Biển số", "")
        brand = event.get("Hãng", "")
        xe_id = plate if plate else f"TRACK-{event.get('Track ID', 0)}"
        event["Xe ID"] = xe_id
        with self.connect() as db:
            with db.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO events (
                        event_id, recorded_at,
                        camera, xe_id, track_id, started_at, ended_at,
                        vehicle_type, brand, brand_score, plate, plate_score,
                        images
                    )
                    VALUES (
                        %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s
                    )
                    """,
                    (
                        event["event_id"],
                        now.replace(tzinfo=None),
                        event.get("Camera", ""),
                        event.get("Xe ID", ""),
                        int(event.get("Track ID", 0)),
                        float(event.get("Xuất hiện (giây)", 0)),
                        float(event.get("Kết thúc (giây)", 0)),
                        event.get("Loại xe", ""),
                        event.get("Hãng", ""),
                        float(event.get("Điểm hãng", 0)),
                        event.get("Biển số", ""),
                        float(event.get("Điểm biển", 0)),
                        json.dumps({"detected": [], "plate": []}, ensure_ascii=False, separators=(",", ":")),
                    ),
                )
        return event

    def update_images(self, event_id, saved):
        with self.connect() as db:
            with db.cursor() as cursor:
                cursor.execute(
                    "UPDATE events SET images = %s WHERE event_id = %s",
                    (
                        json.dumps(saved, ensure_ascii=False, separators=(",", ":")),
                        event_id,
                    ),
                )

    def recent(self, limit=200):
        with self.connect() as db:
            with db.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT event_id, recorded_at, camera, track_id,
                           xe_id,
                           started_at, ended_at, vehicle_type, brand,
                           brand_score, plate, plate_score,
                           images
                    FROM events
                    ORDER BY recorded_at DESC, event_id DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                rows = cursor.fetchall()
        result = []
        for row in rows:
            (
                event_id, recorded_at, camera, track_id, xe_id,
                started_at, ended_at,
                vehicle_type, brand, brand_score, plate, plate_score,
                images,
            ) = row
            item = {
                "event_id": event_id,
                "recorded_at": recorded_at.isoformat() + "+00:00",
                "Camera": camera or "",
                "Track ID": track_id or 0,
                "Xe ID": xe_id or "",
                "Xuất hiện (giây)": started_at or 0,
                "Loại xe": vehicle_type or "",
                "Hãng": brand or "",
                "Điểm hãng": brand_score or 0,
                "Biển số": plate or "",
                "Điểm biển": plate_score or 0,
                "Kết thúc (giây)": ended_at or 0,
            }
            if images:
                item["images"] = json.loads(images) if isinstance(images, str) else images
            result.append(item)
        return result
