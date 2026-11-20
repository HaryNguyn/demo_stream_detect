#!/usr/bin/env python3
"""Remap YOLO logo labels by brand name into a separate directory.

Example:
    python scripts/remap_logo_labels.py --input data/logo_old/labels \
        --output data/logo_v2/labels --new-classes configs/classes_logo.example.yaml \
        --report runtime/evaluation/logo_remap_report

Use --dry-run to inspect mapping and statistics without writing any files.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

# Verify against the taxonomy used to annotate YOUR dataset; override when different.
OLD_NAMES_DEFAULT = [
    "audi", "bmw", "chevrolet", "citroen", "dacia", "fiat", "ford", "honda",
    "hyundai", "kia", "lexus", "mazda", "mercedes", "mitsubishi", "nissan",
    "opel", "peugeot", "renault", "seat", "skoda", "suzuki", "toyota",
    "volkswagen", "volvo",
]
MERGE_TO_OTHER = {
    "opel", "seat", "skoda", "dacia", "citroen", "fiat",
    "chevrolet", "audi", "renault", "volkswagen",
}
UNDECIDED = {"nissan"}
SPLIT_ALIASES = {
    "train": "train", "val": "val", "valid": "val", "validation": "val", "test": "test",
}
SKIP_FILES = {"classes.txt", "labels.txt", "readme.txt"}
SPLIT_ORDER = ["train", "val", "test", "all"]


def load_names(path):
    data = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    names = data.get("names", data) if isinstance(data, dict) else data
    if isinstance(names, dict):
        indexed = {}
        for key, value in names.items():
            if isinstance(key, bool) or str(key) != str(int(key)):
                raise ValueError(f"{path}: class IDs must be integers")
            index = int(key)
            if index in indexed:
                raise ValueError(f"{path}: duplicate class ID {index}")
            indexed[index] = value
        if sorted(indexed) != list(range(len(indexed))):
            raise ValueError(f"{path}: class IDs must be contiguous from zero")
        names = [indexed[i] for i in range(len(indexed))]
    if not isinstance(names, list) or not names:
        raise ValueError(f"{path}: expected a nonempty names list")
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError(f"{path}: class names must be nonempty strings")
    names = [name.strip().lower() for name in names]
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: duplicate class names")
    return names


def build_mapping(old_names, new_names, overrides):
    if set(overrides) - set(old_names):
        raise ValueError(f"Unknown override source(s): {sorted(set(overrides) - set(old_names))}")
    mapping, warnings = {}, []
    for old_id, name in enumerate(old_names):
        if name in overrides:
            target = overrides[name]
        elif name in new_names:
            target = name
        elif name in MERGE_TO_OTHER | UNDECIDED:
            target = "other"
            if name in UNDECIDED:
                warnings.append(f"'{name}' mapped to 'other'; use --map {name}=<class> to override.")
        else:
            raise ValueError(f"No rule for '{name}'; use --map {name}=<class>.")
        if target not in new_names:
            raise ValueError(f"Target '{target}' for '{name}' is not in the new taxonomy")
        mapping[old_id] = new_names.index(target)
    return mapping, warnings


def split_of(rel):
    for part in rel.parts[:-1]:
        if part.lower() in SPLIT_ALIASES:
            return SPLIT_ALIASES[part.lower()]
    return "all"


def collect(root, old_names, new_names, mapping):
    files = sorted(p for p in root.rglob("*")
                   if p.is_file() and p.suffix.lower() == ".txt" and p.name.lower() not in SKIP_FILES)
    if not files:
        raise ValueError(f"No label files under {root}")
    results, errors = {}, []
    stats = defaultdict(lambda: {
        "files": 0, "empty": 0, "old": Counter(), "new": Counter(), "new_images": Counter(),
    })
    for path in files:
        rel = path.relative_to(root)
        st = stats[split_of(rel)]
        st["files"] += 1
        lines, seen = [], set()
        text = path.read_text(encoding="utf-8-sig")
        for number, raw in enumerate(text.splitlines(), 1):
            fields = raw.split()
            if not fields:
                continue
            try:
                if len(fields) != 5:
                    raise ValueError("expected exactly 5 detection fields")
                old_id = int(fields[0])
                if old_id not in mapping:
                    raise ValueError(f"class ID outside 0..{len(old_names) - 1}")
                x, y, w, h = map(float, fields[1:])
                if not all(math.isfinite(v) and 0 <= v <= 1 for v in (x, y, w, h)):
                    raise ValueError("coordinates must be finite and normalized to [0, 1]")
                if w <= 0 or h <= 0:
                    raise ValueError("box width and height must be positive")
            except ValueError as error:
                errors.append(f"{rel}:{number}: {error}")
                continue
            new_id = mapping[old_id]
            lines.append(" ".join([str(new_id)] + fields[1:]))
            st["old"][old_names[old_id]] += 1
            st["new"][new_names[new_id]] += 1
            seen.add(new_id)
        if not text.strip():
            st["empty"] += 1
        for new_id in seen:
            st["new_images"][new_names[new_id]] += 1
        results[rel] = lines
    return results, errors, stats


def make_warnings(stats, names, min_train, min_eval):
    warnings = []
    for split, st in stats.items():
        threshold = min_train if split in ("train", "all") else min_eval
        if split == "all":
            warnings.append("Unrecognized split folders: 'all' uses --min-train as an advisory threshold.")
        for name in names:
            count = st["new"].get(name, 0)
            if count == 0 or count < threshold:
                warnings.append(f"[{split}] '{name}': {count} boxes (threshold {threshold})")
    return warnings


def print_report(old_names, new_names, mapping, stats):
    print("\nMapping (old -> new):")
    for old_id, new_id in mapping.items():
        print(f"  {old_id:2} {old_names[old_id]:12} -> {new_id:2} {new_names[new_id]}")
    for split in SPLIT_ORDER:
        if split not in stats:
            continue
        st = stats[split]
        print(f"\n{split}: {st['files']} files, {st['empty']} empty")
        print(f"  {'class':12} {'boxes':>8} {'images':>8}")
        for name in new_names:
            print(f"  {name:12} {st['new'][name]:8} {st['new_images'][name]:8}")


def report_paths(prefix):
    return [prefix.with_suffix(".json"), prefix.with_suffix(".csv")]


def write_report(prefix, old_names, new_names, mapping, stats, warnings):
    json_path, csv_path = report_paths(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "old_names": old_names, "new_names": new_names,
        "mapping": {old_names[o]: new_names[n] for o, n in mapping.items()},
        "warnings": warnings,
        "splits": {
            split: {"label_files": st["files"], "empty_files": st["empty"],
                    "boxes_old": dict(st["old"]), "boxes_new": dict(st["new"]),
                    "images_new": dict(st["new_images"])}
            for split, st in stats.items()
        },
    }
    with json_path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    with csv_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["split", "class", "boxes", "images"])
        for split in SPLIT_ORDER:
            if split in stats:
                for name in new_names:
                    writer.writerow([split, name, stats[split]["new"][name],
                                     stats[split]["new_images"][name]])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--new-classes", required=True, type=Path)
    parser.add_argument("--old-classes", type=Path, help="Original annotation taxonomy YAML")
    parser.add_argument("--map", action="append", default=[], metavar="OLD=NEW")
    parser.add_argument("--report", type=Path, help="Prefix for JSON/CSV reports")
    parser.add_argument("--min-train", type=int, default=200)
    parser.add_argument("--min-eval", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        if min(args.min_train, args.min_eval) < 0:
            raise ValueError("Minimum counts must be nonnegative")
        old_names = load_names(args.old_classes) if args.old_classes else OLD_NAMES_DEFAULT
        new_names = load_names(args.new_classes)
        overrides = {}
        for item in args.map:
            key, separator, value = item.partition("=")
            key, value = key.strip().lower(), value.strip().lower()
            if not separator or not key or not value or key in overrides:
                raise ValueError(f"Invalid or duplicate --map: {item}")
            overrides[key] = value
        mapping, warnings = build_mapping(old_names, new_names, overrides)
        root = args.input.resolve()
        if not root.is_dir():
            raise ValueError(f"Input directory not found: {root}")
        out = args.output.resolve() if args.output else None
        if out is not None:
            if out == root or root in out.parents or out in root.parents:
                raise ValueError("Output must be separate from input, not its ancestor or descendant")
            if out.exists():
                raise ValueError(f"Output must be a NEW directory: {out}")
        if not args.dry_run and out is None:
            raise ValueError("--output is required unless --dry-run")
        if args.report:
            for path in report_paths(args.report):
                resolved = path.resolve()
                if resolved == root or root in resolved.parents:
                    raise ValueError("Reports must be outside the input dataset")
                if resolved.exists():
                    raise ValueError(f"Report already exists: {path}")
                if out and (resolved == out or resolved in out.parents):
                    raise ValueError("Report path conflicts with output directory")
        results, errors, stats = collect(root, old_names, new_names, mapping)
        if errors:
            raise ValueError(f"{len(errors)} invalid label lines; nothing written:\n" +
                             "\n".join(errors[:20]))
        warnings += make_warnings(stats, new_names, args.min_train, args.min_eval)
        print_report(old_names, new_names, mapping, stats)
        for warning in warnings:
            print(f"WARNING: {warning}")
        if args.dry_run:
            print("\nDry run: no files written (including reports).")
            return
        out.mkdir(parents=True, exist_ok=False)
        for rel, lines in results.items():
            dest = out / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        (out / "classes.yaml").write_text(
            yaml.safe_dump({"names": new_names}, sort_keys=False), encoding="utf-8")
        if args.report:
            write_report(args.report, old_names, new_names, mapping, stats, warnings)
        print(f"\nWrote {len(results)} label files + classes.yaml to {out}")
    except (ValueError, TypeError, OSError, yaml.YAMLError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
