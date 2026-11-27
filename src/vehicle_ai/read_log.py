"""Per-run image latency records shared by the web UI and the image benchmark."""
from contextvars import ContextVar
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from threading import Lock
from time import perf_counter
from uuid import UUID, uuid4


_active = ContextVar("image_stage_times", default=None)
_write_lock = Lock()


def run_id(value=None):
    try:
        return str(UUID(str(value)))
    except ValueError:
        return str(uuid4())


def timed_stage(name, function, *args, **kwargs):
    stages = _active.get()
    if stages is None:
        return function(*args, **kwargs)
    started = perf_counter()
    try:
        return function(*args, **kwargs)
    finally:
        stage = stages.setdefault(name, {"ms": 0.0, "calls": 0})
        stage["ms"] += (perf_counter() - started) * 1000
        stage["calls"] += 1


def logged_read(batch, filename, recognize, *, input_kind="auto"):
    started = perf_counter()
    stages = {}
    token = _active.set(stages)
    record = {"run_id": batch, "filename": filename, "input_kind": input_kind,
              "started_at": datetime.now(timezone.utc).isoformat()}
    try:
        result = recognize()
        record.update(status="ok", route=result.get("input_route"), width=result.get("width"),
                      height=result.get("height"), timings_ms=result.get("timings_ms", {}),
                      summary=result.get("summary", {}))
        return result
    except Exception as error:
        record.update(status="error", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        _active.reset(token)
        record["elapsed_ms"] = round((perf_counter() - started) * 1000, 2)
        record["stages"] = {k: {**v, "ms": round(v["ms"], 2)} for k, v in stages.items()}
        path = Path("runtime/read_logs") / f"{batch}.jsonl"
        try:
            with _write_lock:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            logging.getLogger(__name__).exception("Cannot append image timing to %s", path)
