#!/usr/bin/env python
"""Collect Overall rows from MMIU perf summaries into one CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_DIR / "outputs" / "mmiu"
DEFAULT_OUTPUT = PROJECT_DIR / "outputs" / "mmiu" / "mmiu_perf_summary.csv"
METRIC_FIELDS = [
    "duration_s",
    "request_throughput",
    "output_token_throughput",
    "total_token_throughput",
    "mean_latency_s",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default=str(DEFAULT_ROOT),
        help="Directory containing *_mmiu_perf_summary.csv files.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output CSV path.",
    )
    return parser.parse_args()


def read_overall_row(path: Path) -> dict[str, str] | None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("task") == "Overall":
                row["source_path"] = str(path)
                return row
    return None


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()

    if not root.exists():
        raise SystemExit(f"root not found: {root}")

    rows = []
    for path in sorted(root.glob("*_mmiu_perf_summary.csv")):
        if path.resolve() == output:
            continue
        row = read_overall_row(path)
        if row is not None:
            rows.append(row)

    if not rows:
        raise SystemExit(f"no Overall rows found under {root}")

    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "task", *METRIC_FIELDS, "source_path"]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    print(f"saved {output}")
    print(f"rows={len(rows)}")


if __name__ == "__main__":
    main()
