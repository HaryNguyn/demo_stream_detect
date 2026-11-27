"""Clock-paced preview with a bounded, latest-frame queue for inference."""
from dataclasses import dataclass
from queue import Empty, Full, Queue
from threading import Event, Thread
from time import monotonic


@dataclass
class PlaybackProgress:
    frames_read: int = 0
    frames_eligible: int = 0


def realtime_frames(capture, fps, total, max_seconds, on_frame, should_stop=None, *,
                    frame_stride=1, progress=None):
    if progress is None:
        progress = PlaybackProgress()
    pending = Queue(maxsize=1)
    finished = Event()
    cancelled = Event()
    errors = []

    def stopped():
        return cancelled.is_set() or (should_stop and should_stop())

    def produce():
        try:
            start = None
            index = 0
            while not stopped() and (not max_seconds or index / fps < max_seconds):
                delay = start + index / fps - monotonic() if start is not None else 0
                while delay > 0 and not stopped():
                    cancelled.wait(min(delay, 0.05))
                    delay = start + index / fps - monotonic()
                if stopped():
                    break
                ok, frame = capture.read()
                if not ok:
                    break
                if start is None:
                    # Connection/model warmup must not become a playback backlog.
                    start = monotonic()
                progress.frames_read += 1
                # Only frames eligible for inference can replace a pending inference frame.
                if index % frame_stride == 0:
                    progress.frames_eligible += 1
                    try:
                        pending.put_nowait((index, frame))
                    except Full:
                        try:
                            pending.get_nowait()
                        except Empty:
                            pass
                        pending.put_nowait((index, frame))
                # Skip stale preview frames if decoding/encoding falls behind.
                if on_frame and index >= int((monotonic() - start) * fps) - 1:
                    on_frame(index + 1, total, frame)
                index += 1
        except Exception as error:
            errors.append(error)
        finally:
            finished.set()

    worker = Thread(target=produce, daemon=True)
    worker.start()
    try:
        while not stopped():
            try:
                yield pending.get(timeout=0.05)
            except Empty:
                if finished.is_set():
                    break
        if errors:
            raise errors[0]
    finally:
        cancelled.set()
        worker.join()
