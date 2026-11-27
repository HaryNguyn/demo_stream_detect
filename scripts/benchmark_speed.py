"""Benchmark recognition presets on the same video source.

Usage:
    python scripts/benchmark_speed.py --video path/to/video.mp4 --max-seconds 10
"""
import argparse
import logging
import statistics
import time

logging.getLogger("ultralytics").setLevel(logging.ERROR)

from ultralytics.utils import LOGGER  # noqa: E402

LOGGER.disabled = True

from vehicle_ai.config import apply_speed_preset, load_config  # noqa: E402
from vehicle_ai.inference import optional_yolo  # noqa: E402
from vehicle_ai.model_loader import load_recognizer  # noqa: E402
from vehicle_ai.pipeline import analyze_video  # noqa: E402


def summarize(values):
    if not values:
        return "-"
    return f"{statistics.median(values):.1f}/{max(values):.1f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--config", default="configs/vietnam-ocr.yaml")
    parser.add_argument("--presets", nargs="+", default=["accurate", "balanced", "fast"])
    parser.add_argument("--max-seconds", type=float, default=10)
    parser.add_argument("--detector-only", action="store_true",
                        help="Skip logo/plate/OCR models to isolate detector cost.")
    args = parser.parse_args()

    base = load_config(args.config)
    logo = plate = recognizer = None
    if not args.detector_only:
        logo = optional_yolo(base.logo_weights) if base.logo_weights else None
        plate = optional_yolo(base.plate_weights) if base.plate_weights else None
        recognizer = load_recognizer(base)

    print(f"video={args.video} seconds={args.max_seconds} detector_only={args.detector_only}")
    print("preset  frames  wall_s  fps   median/max ms (total/vehicle/crops/ocr)")
    for preset in args.presets:
        config = apply_speed_preset(load_config(args.config), preset)
        samples = []
        stats = {}

        def on_diagnostic(info):
            timing = info.get("timings_ms")
            if timing:
                samples.append(timing)

        started = time.perf_counter()
        analyze_video(
            args.video, "benchmark", config.vehicle_weights, logo, plate, recognizer,
            config=config, max_seconds=args.max_seconds, collect_results=False,
            on_diagnostic=on_diagnostic, on_stats=stats.update, realtime=False,
        )
        elapsed = max(time.perf_counter() - started, 1e-6)
        frames = stats.get("frames", 0)
        columns = [summarize([item[name] for item in samples]) for name in
                   ("total", "vehicle", "plate_and_crops", "ocr")]
        print(f"{preset:<8}{frames:>6}{elapsed:>8.2f}{frames / elapsed:>7.2f}  "
              + "  ".join(columns))


if __name__ == "__main__":
    main()
