"""Event repository interface and MariaDB factory."""
from typing import Protocol

from .mariadb import MariaDBEventStore

__all__ = ["EventRepository", "MariaDBEventStore", "create_event_store"]


class EventRepository(Protocol):
    def append(self, row: dict) -> dict: ...
    def update_images(self, event_id: str, saved: dict) -> None: ...
    def recent(self, limit: int = 200) -> list[dict]: ...


def create_event_store() -> EventRepository:
    """Persist all application events in MariaDB."""
    return MariaDBEventStore()
