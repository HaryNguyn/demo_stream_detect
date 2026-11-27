"""Prepare and summarize the 10/30-minute x 144p..1080p video test matrix.

Subcommands:
    configs  generate one valid YAML config per resolution
    plan     discover videos, write manifest.csv and runnable command files
    report   aggregate run.json + evaluate-events reports into matrix.csv/json
"""
import argparse
import csv
import json
from pathlib import Path
import shlex
import sys

import yaml

from vehicle_ai.config import load_config
from vehicle_ai.measurement import (
    RESOLUTION_SIZES, VIDEO_SUFFIXES, parse_video_name, rebase_config_paths,
    resolution_overrides,
)

HEIGHTS = (144, 240, 360, 480, 720, 1080)
RUN_FIELDS = (
    "frames", "source_fps", "content_seconds", "elapsed_seconds", "events",
    "throughput_fps_including_load", "frame_size",
)
EVAL_FIELDS = (
    "ground_truth_events", "predicted_events", "matched_events", "missed_events",
    "extra_events", "event_recall", "event_precision", "event_f1",
    "plate_end_to_end_recall", "plate_cer_on_matched", "confirmation_rate",
)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _ps_quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def cmd_configs(args):
    base = Path(args.base).resolve()
    data = yaml.safe_load(base.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Config gốc phải là YAML mapping.")
    target = Path(args.output_dir).resolve()
    target.mkdir(parents=True, exist_ok=True)
    generated = []
    for height in args.heights or HEIGHTS:
        overrides = resolution_overrides(data, height, args.reference_height)
        payload = rebase_config_paths({**data, **overrides}, base.parent, target)
        path = target / f"{args.prefix}{height}p.yaml"
        if path.exists() and not args.force:
            raise FileExistsError(f"Đã tồn tại {path}; dùng --force để ghi lại.")
        path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
                        encoding="utf-8")
        load_config(path)  # validate before the long runs
        generated.append({
            "height": height,
            "resolution": f"{height}p",
            "size": list(RESOLUTION_SIZES.get(f"{height}p", (0, height))),
            "config": str(path),
            "overrides": overrides,
        })
    _write_json(target / "configs.json", {
        "base": str(base), "reference_height": args.reference_height,
        "note": "Chỉ scale ngưỡng pixel; detector_imgsz giữ nguyên vì là input model.",
        "configs": generated,
    })
    print(f"Đã tạo {len(generated)} cấu hình trong {target}")


def cmd_plan(args):
    videos = sorted(path for path in Path(args.videos).rglob("*")
                    if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES)
    truth_dir = Path(args.truth_dir).resolve()
    config_dir = Path(args.config_dir).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    plans = []
    for video in videos:
        info = parse_video_name(video)
        if not info:
            print(f"Bỏ qua {video.name}: tên không có '<số>min_<số>p'.")
            continue
        stem = info["source_id"]
        candidates = (truth_dir / f"{stem}.truth.jsonl", truth_dir / f"{stem}.jsonl")
        truth = next((path for path in candidates if path.is_file()), None)
        if truth is None:
            raise FileNotFoundError(f"Thiếu ground truth cho {video.name} trong {truth_dir}.")
        config = config_dir / f"{args.prefix}{info['resolution']}.yaml"
        if not config.is_file():
            raise FileNotFoundError(f"Thiếu cấu hình {config}; chạy lệnh 'configs' trước.")
        run_dir = output / f"{stem}-run"
        merged = run_dir / "eval-merged.json"
        raw = run_dir / "eval-raw.json"
        rows.append({
            "source_id": stem, "video": str(video.resolve()), "truth": str(truth),
            "config": str(config), "resolution": info["resolution"],
            "duration_min": info["duration_min"], "run_dir": str(run_dir),
            "report": str(merged), "report_raw": str(raw),
        })
        run_cmd = [
            "vehicle-ai", "run", "--source", str(video.resolve()), "--source-id", stem,
            "--camera", args.camera, "--config", str(config), "--max-seconds", "0",
            "--output", str(run_dir),
        ]
        if args.roi_file:
            run_cmd.extend(["--roi-file", str(Path(args.roi_file).resolve())])
        merged_cmd = [
            "vehicle-ai", "evaluate-events", "--truth", str(truth),
            "--predictions", str(run_dir / "events.merged.jsonl"), "--output", str(merged),
        ]
        raw_cmd = [
            "vehicle-ai", "evaluate-events", "--truth", str(truth),
            "--predictions", str(run_dir / "events.jsonl"), "--output", str(raw),
        ]
        plans.append({"stem": stem, "commands": [run_cmd, merged_cmd, raw_cmd]})
    if not rows:
        raise ValueError(f"Không tìm thấy video hợp lệ trong {args.videos}.")
    manifest = output / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    ps_lines, shell_lines = [], []
    for plan in plans:
        ps_lines.append("& " + " ".join(_ps_quote(part) for part in plan["commands"][0]))
        ps_lines.append("if ($LASTEXITCODE -ne 0) { throw 'run failed: "
                        + plan["stem"] + "' }")
        shell_lines.append(" ".join(shlex.quote(part) for part in plan["commands"][0]))
        for command in plan["commands"][1:]:
            ps_lines.append("& " + " ".join(_ps_quote(part) for part in command))
            shell_lines.append(" ".join(shlex.quote(part) for part in command))
    (output / "commands.ps1").write_text(
        "$ErrorActionPreference = 'Stop'\n" + "\n".join(ps_lines) + "\n", encoding="utf-8")
    (output / "commands.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n" + "\n".join(shell_lines) + "\n",
        encoding="utf-8")
    _write_json(output / "plan.json", {
        "videos_root": str(Path(args.videos).resolve()), "rows": rows,
        "note": "Chạy commands.ps1, sau đó chạy 'report' để gộp kết quả.",
    })
    print(f"Kế hoạch {len(rows)} video: {manifest}")


def cmd_report(args):
    with Path(args.manifest).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    entries = []
    for row in rows:
        entry = {
            "source_id": row["source_id"], "resolution": row["resolution"],
            "duration_min": row["duration_min"], "status": "pending",
        }
        run_file = Path(row["run_dir"]) / "run.json"
        report_file = Path(row["report"])
        if not run_file.is_file():
            entry["status"] = "missing_run"
            entries.append(entry)
            continue
        run = json.loads(run_file.read_text(encoding="utf-8"))
        entry["status"] = run.get("status", "unknown")
        for name in RUN_FIELDS:
            entry[name] = run.get(name)
        config = run.get("config") or {}
        for name in ("model_version", "detector_imgsz", "min_vehicle_width",
                     "min_plate_width", "quality_target_width", "quality_target_height"):
            entry[f"config_{name}"] = config.get(name)
        if report_file.is_file():
            report = json.loads(report_file.read_text(encoding="utf-8"))
            for name in EVAL_FIELDS:
                entry[name] = report.get(name)
            for tag in (row["resolution"], f"{row['duration_min']}min"):
                group = (report.get("tag_groups") or {}).get(tag) or {}
                prefix = tag.replace("-", "_")
                entry[f"tag_{prefix}_recall"] = group.get("event_recall")
                entry[f"tag_{prefix}_plate_e2e"] = group.get("plate_end_to_end_recall")
        else:
            entry["status"] = "missing_report"
        entries.append(entry)

    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for entry in entries for key in entry))
    with (output_dir / "matrix.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for entry in entries:
            writer.writerow({key: json.dumps(value, ensure_ascii=False)
                             if isinstance(value, (dict, list)) else value
                             for key, value in entry.items()})
    _write_json(output_dir / "matrix.json", {"runs": entries})
    print(f"Báo cáo: {output_dir / 'matrix.csv'}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    configs = sub.add_parser("configs", help="Tạo config riêng cho từng độ phân giải.")
    configs.add_argument("--base", default="configs/vietnam-ocr.yaml")
    configs.add_argument("--output-dir", default="configs/resolutions")
    configs.add_argument("--reference-height", type=int, default=720)
    configs.add_argument("--heights", nargs="+", type=int, default=list(HEIGHTS))
    configs.add_argument("--prefix", default="res")
    configs.add_argument("--force", action="store_true")
    plan = sub.add_parser("plan", help="Tạo manifest và lệnh chạy cho ma trận video.")
    plan.add_argument("--videos", required=True)
    plan.add_argument("--truth-dir", required=True)
    plan.add_argument("--config-dir", default="configs/resolutions")
    plan.add_argument("--prefix", default="res")
    plan.add_argument("--output", required=True)
    plan.add_argument("--camera", default="CAM-01")
    plan.add_argument("--roi-file")
    report = sub.add_parser("report", help="Gộp run.json và báo cáo đánh giá.")
    report.add_argument("--manifest", required=True)
    report.add_argument("--output", required=True)
    return parser


def main():
    args = build_parser().parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    {"configs": cmd_configs, "plan": cmd_plan, "report": cmd_report}[args.command](args)


if __name__ == "__main__":
    main()
