"""Event repository interface and backend selection."""
from pathlib import Path
from typing import Protocol

from ..settings import get_setting, project_dir
from .mariadb import MariaDBEventStore
from .sqlite import SQLiteEventStore

__all__ = ["EventRepository", "MariaDBEventStore", "SQLiteEventStore", "create_event_store"]


class EventRepository(Protocol):
    def append(self, row: dict) -> dict: ...
    def update_images(self, event_id: str, saved: dict) -> None: ...
    def recent(self, limit: int = 200) -> list[dict]: ...


def create_event_store(path: str | Path | None = None) -> EventRepository:
    """An explicit path selects SQLite, useful for local tools and tests."""
    backend = get_setting("EVENT_STORE", "sqlite").lower()
    if path is not None or backend == "sqlite":
        return SQLiteEventStore(path or project_dir() / "runtime" / "events.sqlite3")
    if backend == "mariadb":
        return MariaDBEventStore()
    raise ValueError("EVENT_STORE phải là sqlite hoặc mariadb.")
