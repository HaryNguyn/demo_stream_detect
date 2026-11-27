"""Measure the image pipeline on cumulative batches (e.g. 50/500/5000/10000).

Usage:
    python scripts/measure_images.py --images data/images --counts 50 500 5000 10000 \
        --truth data/images.truth.jsonl --output runtime/measure-images
"""
import argparse
import json
from pathlib import Path
import random
import sys
import time

from vehicle_ai.measurement import (
    IMAGE_SUFFIXES, compact_result, read_jsonl, summarize_images,
)
from vehicle_ai.plate_image import DEFAULT_OPTIONS, recognize_image


def parse_options(items):
    options = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Tùy chọn phải dạng KEY=VALUE: {item}")
        key, value = item.split("=", 1)
        if key not in DEFAULT_OPTIONS:
            raise ValueError(f"Khóa không hợp lệ: {key}")
        options[key] = value.strip()
    return options


def discover_images(root, shuffle=False, seed=0):
    images = sorted(path for path in Path(root).rglob("*")
                    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if shuffle:
        random.Random(seed).shuffle(images)
    return images


def load_truth(path):
    truth = {}
    if not path:
        return truth
    for row in read_jsonl(path):
        key = row.get("filename") or row.get("image")
        if not key:
            raise ValueError("Ảnh trong ground truth thiếu filename.")
        truth[Path(str(key)).as_posix()] = row
    return truth


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, help="Thư mục chứa ảnh.")
    parser.add_argument("--output", required=True, help="Thư mục mới; không ghi đè kết quả cũ.")
    parser.add_argument("--counts", nargs="+", type=int, default=[50, 500, 5000, 10000])
    parser.add_argument("--truth", help="JSONL: filename, plate, vehicle_type, brand, tags.")
    parser.add_argument("--config", default="configs/vietnam-ocr.yaml")
    parser.add_argument("--warmup", type=int, default=2, help="Số ảnh đầu không tính vào độ trễ.")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=0, help="0 = dùng số ảnh lớn nhất trong counts.")
    parser.add_argument("--option", action="append", default=[], metavar="KEY=VALUE",
                        help="Ghi đè tùy chọn ảnh, ví dụ min_vehicle_width=24.")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    counts = sorted(set(args.counts))
    if not counts or counts[0] < 1:
        raise ValueError("counts phải chứa số nguyên dương.")
    if args.warmup < 0:
        raise ValueError("warmup không được âm.")
    options = dict(DEFAULT_OPTIONS)
    options.update({"mode": "yaml", "config": str(Path(args.config).resolve())})
    options.update(parse_options(args.option))

    images = discover_images(args.images, shuffle=args.shuffle, seed=args.seed)
    limit = max(counts)
    if args.max_images:
        limit = min(limit, args.max_images)
    images = images[:limit]
    if len(images) < max(counts):
        print(f"Cảnh báo: chỉ có {len(images)} ảnh, nhỏ hơn {max(counts)}.", file=sys.stderr)

    truth = load_truth(args.truth)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "manifest.json").write_text(json.dumps({
        "image_root": str(Path(args.images).resolve()),
        "images_used": [str(path) for path in images],
        "counts": counts, "warmup": args.warmup, "shuffle": args.shuffle, "seed": args.seed,
        "options": options, "truth": str(Path(args.truth).resolve()) if args.truth else "",
        "note": "Kết quả đã bỏ base64 ảnh; chỉ giữ nhãn và điểm để chấm hàng loạt.",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    image_root = Path(args.images).resolve()
    started = time.perf_counter()
    with (output / "results.jsonl").open("w", encoding="utf-8") as stream:
        for index, path in enumerate(images, 1):
            if index % 500 == 0:
                print(f"Đã xử lý {index}/{len(images)} ảnh...", flush=True)
            row = {
                "filename": path.resolve().relative_to(image_root).as_posix(),
                "path": str(path),
            }
            try:
                row.update(compact_result(recognize_image(path.read_bytes(), options)))
            except Exception as error:
                row["error"] = f"{type(error).__name__}: {error}"
            row["elapsed_total_s"] = round(time.perf_counter() - started, 4)
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    rows = read_jsonl(output / "results.jsonl")
    summary = {
        "purpose": "image throughput and accuracy per cumulative batch",
        "counts": {str(count): summarize_images(rows[:count], truth, args.warmup) for count in counts},
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    for count, metrics in summary["counts"].items():
        print(f"{count:>6} ảnh | {metrics['images_per_second'] or 0:6.3f} img/s | "
              f"p95 {metrics['p95_ms'] or 0:7.1f} ms | lỗi {metrics['errors']}")
    print(f"Kết quả: {output}")


if __name__ == "__main__":
    main()
