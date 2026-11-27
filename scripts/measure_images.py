"""Measure the image pipeline on cumulative batches (e.g. 50/500/5000/10000).

Usage:
    python scripts/measure_images.py --images data/images --counts 50 500 5000 10000 \
        --truth data/images.truth.jsonl --output runtime/measure-images
"""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import sys
import time

from vehicle_ai.measurement import (
    IMAGE_SUFFIXES, compact_result, read_jsonl, summarize_images,
)
from vehicle_ai.plate_image import DEFAULT_OPTIONS, recognize_image
from vehicle_ai.read_log import logged_read, run_id


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


def prefetch_images(executor, paths, limit):
    remaining = iter(paths)
    pending = deque((path, executor.submit(path.read_bytes))
                    for _, path in zip(range(limit), remaining))
    while pending:
        path, future = pending.popleft()
        next_path = next(remaining, None)
        if next_path is not None:
            pending.append((next_path, executor.submit(next_path.read_bytes)))
        yield path, future


def load_truth(path, images=(), root="."):
    truth = {}
    if not path:
        return truth
    if Path(path).suffix.lower() == ".xlsx":
        from vehicle_ai.excel_truth import read_excel_truth

        rows = read_excel_truth(path)
        if "filename" not in rows[0]:
            names = sorted((image.relative_to(root).as_posix() for image in images), key=str.lower)
            if len(names) != len(rows):
                raise ValueError("Excel chỉ có PLATE_NUMBER: số dòng phải bằng số ảnh trong thư mục.")
            return dict(zip(names, rows))
    else:
        rows = read_jsonl(path)
    for row in rows:
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
    parser.add_argument("--io-workers", type=int, choices=range(1, 9), default=2,
                        help="Số luồng đọc tệp trước khi đưa ảnh vào model.")
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
    truth = load_truth(args.truth, images, args.images)
    limit = max(counts)
    if args.max_images:
        limit = min(limit, args.max_images)
    images = images[:limit]
    if len(images) < max(counts):
        print(f"Cảnh báo: chỉ có {len(images)} ảnh, nhỏ hơn {max(counts)}.", file=sys.stderr)

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    batch = run_id()
    print(f"Log: runtime/read_logs/{batch}.jsonl", flush=True)
    (output / "manifest.json").write_text(json.dumps({
        "image_root": str(Path(args.images).resolve()),
        "images_used": [str(path) for path in images],
        "counts": counts, "warmup": args.warmup, "shuffle": args.shuffle, "seed": args.seed,
        "io_workers": args.io_workers,
        "run_id": batch,
        "options": options, "truth": str(Path(args.truth).resolve()) if args.truth else "",
        "note": "Kết quả đã bỏ base64 ảnh; chỉ giữ nhãn và điểm để chấm hàng loạt.",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    image_root = Path(args.images).resolve()
    started = time.perf_counter()
    with (
        ThreadPoolExecutor(max_workers=args.io_workers) as reader,
        (output / "results.jsonl").open("w", encoding="utf-8") as stream,
    ):
        for index, (path, payload) in enumerate(prefetch_images(reader, images, args.io_workers * 2), 1):
            row = {
                "filename": path.resolve().relative_to(image_root).as_posix(),
                "path": str(path),
            }
            try:
                result = logged_read(batch, row["filename"], lambda: recognize_image(
                    payload.result(), options, preview="none"), input_kind=options["input_kind"])
                row.update(compact_result(result))
            except Exception as error:
                row["error"] = f"{type(error).__name__}: {error}"
            row["elapsed_total_s"] = round(time.perf_counter() - started, 4)
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            if index % 500 == 0:
                print(f"Đã xử lý {index}/{len(images)} ảnh...", flush=True)

    rows = read_jsonl(output / "results.jsonl")
    summary = {
        "purpose": "image throughput and accuracy per cumulative batch",
        "counts": {str(count): summarize_images(rows[:count], truth, args.warmup) for count in counts},
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    for count, metrics in summary["counts"].items():
        print(f"{count:>6} ảnh | {metrics['images_per_second'] or 0:6.3f} img/s | "
              f"TB {metrics['mean_ms'] or 0:7.1f} ms | "
              f"p95 {metrics['p95_ms'] or 0:7.1f} ms | lỗi {metrics['errors']}")
    print(f"Kết quả: {output}")


if __name__ == "__main__":
    main()
