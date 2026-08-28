#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Run MMIU inference against a vLLM OpenAI-compatible server."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI
from PIL import Image


TASKS_WITH_QUESTION_FIRST = {
    "person_reid",
    "multiple_image_captioning",
    "spot_the_similarity",
    "face_retrieval",
    "sketch2image_retrieval",
    "handwritten_retrieval",
    "spot_the_diff",
    "image2image_retrieval",
    "vehicle_retrieval",
    "text2image_retrieval",
    "general_action_recognition",
    "video_captioning",
    "next_img_prediction",
    "temporal_ordering",
    "meme_vedio_understanding",
    "action_quality_assessment",
    "temporal_localization",
    "mevis",
    "ravens_progressive_matrices",
    "threed_indoor_recognition",
    "point_tracking",
    "threed_cad_recognition",
    "single_object_tracking",
}

PERF_FIELDS = [
    "duration_s",
    "request_throughput",
    "output_token_throughput",
    "total_token_throughput",
    "mean_latency_s",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--target-tasks", default="")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--tokens-per-patch", type=int, default=256)
    parser.add_argument("--context-limit", type=int, default=0)
    parser.add_argument("--reuse", action="store_true")
    return parser.parse_args()


def read_model_hints(model_path: str, default_tokens_per_patch: int) -> tuple[int, object]:
    if not model_path:
        return default_tokens_per_patch, "Unknown"
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return default_tokens_per_patch, "Unknown"
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return default_tokens_per_patch, "Unknown"
    tokens = int(data.get("target_tokens", default_tokens_per_patch))
    arch = data.get("llm_config", {}).get("architectures") or data.get("architectures") or "Unknown"
    return tokens, arch


def usage_value(usage: object, key: str) -> int | None:
    if usage is None:
        return None
    if isinstance(usage, dict):
        value = usage.get(key)
    else:
        value = getattr(usage, key, None)
    return int(value) if value is not None else None


def summarize_perf(records: list[dict[str, Any]], model: str, task: str) -> dict[str, Any]:
    successful = [r for r in records if r.get("success")]
    failed = len(records) - len(successful)
    if not records:
        return {
            "model": model,
            "task": task,
            "completed": 0,
            "failed": 0,
            **{field: None for field in PERF_FIELDS},
            "input_tokens": 0,
            "output_tokens": 0,
        }

    first_start = min(float(r["start_time"]) for r in records)
    last_end = max(float(r["end_time"]) for r in records)
    duration = max(last_end - first_start, 1e-9)
    input_tokens = sum(int(r.get("input_tokens") or 0) for r in successful)
    output_tokens = sum(int(r.get("output_tokens") or 0) for r in successful)
    latencies = [float(r["latency_s"]) for r in successful if r.get("latency_s") is not None]
    return {
        "model": model,
        "task": task,
        "completed": len(successful),
        "failed": failed,
        "duration_s": duration,
        "request_throughput": len(successful) / duration,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "output_token_throughput": output_tokens / duration,
        "total_token_throughput": (input_tokens + output_tokens) / duration,
        "mean_latency_s": statistics.fmean(latencies) if latencies else None,
    }


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def fmt_metric(value: object, digits: int = 3) -> str:
    return "null" if value is None else f"{float(value):.{digits}f}"


def write_perf_outputs(task_output_dir: Path, model: str, task: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    requests_path = task_output_dir / "perf_requests.jsonl"
    with requests_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = summarize_perf(records, model, task)
    summary["perf_requests_path"] = str(requests_path)
    write_json(task_output_dir / "perf_summary.json", summary)
    return summary


def load_task_perf(task_output_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    requests_path = task_output_dir / "perf_requests.jsonl"
    summary_path = task_output_dir / "perf_summary.json"
    records: list[dict[str, Any]] = []
    if requests_path.exists():
        with requests_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None
    return records, summary


def summarize_overall_from_tasks(model: str, task_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    duration = sum(float(item.get("duration_s") or 0.0) for item in task_summaries)
    completed = sum(int(item.get("completed") or 0) for item in task_summaries)
    failed = sum(int(item.get("failed") or 0) for item in task_summaries)
    input_tokens = sum(int(item.get("input_tokens") or 0) for item in task_summaries)
    output_tokens = sum(int(item.get("output_tokens") or 0) for item in task_summaries)
    weighted_latency_sum = sum(
        float(item["mean_latency_s"]) * int(item.get("completed") or 0)
        for item in task_summaries
        if item.get("mean_latency_s") is not None
    )
    mean_latency = weighted_latency_sum / completed if completed else None
    return {
        "model": model,
        "task": "Overall",
        "completed": completed,
        "failed": failed,
        "duration_s": duration if task_summaries else None,
        "request_throughput": completed / duration if duration > 0 else None,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "output_token_throughput": output_tokens / duration if duration > 0 else None,
        "total_token_throughput": (input_tokens + output_tokens) / duration if duration > 0 else None,
        "mean_latency_s": mean_latency,
    }


def write_overall_perf(output_dir: Path, model: str, records: list[dict[str, Any]], task_summaries: list[dict[str, Any]]) -> None:
    if not task_summaries and not records:
        return

    overall = summarize_overall_from_tasks(model, task_summaries)
    rows = [*sorted(task_summaries, key=lambda item: item["task"]), overall]

    csv_path = output_dir / f"{model}_mmiu_perf_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "task", *PERF_FIELDS])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in ["model", "task", *PERF_FIELDS]})

    write_json(output_dir / f"{model}_mmiu_perf_summary.json", rows)
    print(f"saved {csv_path}")


def best_resolution_tiles(width: int, height: int, max_num: int = 12) -> int:
    aspect_ratio = width / height
    best_fit: tuple[int, int] | None = None
    min_diff = float("inf")
    for i in range(1, max_num + 1):
        for j in range(1, max_num + 1):
            patches = i * j
            if not 1 <= patches <= max_num:
                continue
            diff = abs(aspect_ratio - i / j)
            if diff < min_diff or (diff == min_diff and best_fit and patches > best_fit[0] * best_fit[1]):
                min_diff = diff
                best_fit = (i, j)
    if best_fit is None:
        return 13
    return best_fit[0] * best_fit[1] + 1


def build_prompt(task_name: str, task_data: dict) -> str:
    input_layer = task_data.get("input", task_data)
    if not isinstance(input_layer, dict):
        input_layer = task_data

    context = input_layer.get("context", "")
    question = input_layer.get("question", "")
    options_text = task_data.get("options", input_layer.get("options", ""))
    if options_text and options_text not in context:
        context = context + "\n" + options_text

    current_task_type = task_data.get("task", task_name)
    if current_task_type in TASKS_WITH_QUESTION_FIRST:
        prompt = question + "\n" + context
    else:
        prompt = context + "\n" + question
    return prompt + "\nPlease answer the option directly(e.g. A, B, C, D):"


def get_image_list(task_data: dict) -> list[str]:
    input_layer = task_data.get("input", task_data)
    if not isinstance(input_layer, dict):
        input_layer = task_data
    key = "input_image_path" if "input_image_path" in input_layer else "image"
    images = input_layer.get(key, [])
    if isinstance(images, str):
        return [images]
    return list(images)


def select_images(
    task_dir: Path,
    img_list: list[str],
    prompt: str,
    tokens_per_patch: int,
    context_limit: int,
) -> list[Path] | None:
    image_paths = []
    for image_path in img_list:
        full_img_path = task_dir / Path(image_path).name
        if not full_img_path.exists():
            return None
        image_paths.append(full_img_path)

    if not image_paths:
        return None

    try:
        with Image.open(image_paths[0]) as img:
            width, height = img.size
        single_image_token_cost = best_resolution_tiles(width, height) * tokens_per_patch
    except Exception:
        single_image_token_cost = 13 * tokens_per_patch

    estimated_text_tokens = len(prompt) // 2
    available_tokens = context_limit - estimated_text_tokens - 1000
    total_image_tokens = single_image_token_cost * len(image_paths)
    if total_image_tokens <= available_tokens:
        return image_paths

    max_images_allowed = max(1, int(available_tokens // single_image_token_cost))
    indices = np.linspace(0, len(image_paths) - 1, max_images_allowed).astype(int)
    return [image_paths[i] for i in indices]


def run_task(
    client: OpenAI,
    task_name: str,
    args: argparse.Namespace,
    tokens_per_patch: int,
    context_limit: int,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    dataset_dir = Path(args.dataset_dir)
    task_dir = dataset_dir / task_name
    json_path = dataset_dir / f"{task_name}.json"
    if not json_path.exists():
        print(f"skip {task_name}: missing {json_path}")
        return [], None

    task_output_dir = Path(args.output_dir) / task_name / args.model
    task_output_dir.mkdir(parents=True, exist_ok=True)
    output_path = task_output_dir / "metadata_info.json"
    if args.reuse and output_path.exists():
        print(f"skip {task_name}: existing {output_path}")
        return load_task_perf(task_output_dir)

    data_all = json.loads(json_path.read_text(encoding="utf-8"))
    result = []
    perf_records: list[dict[str, Any]] = []
    print(f"[{task_name}] samples={len(data_all)}")

    for idx, task_data in enumerate(data_all):
        prompt = build_prompt(task_name, task_data)
        image_paths = select_images(task_dir, get_image_list(task_data), prompt, tokens_per_patch, context_limit)
        if image_paths is None:
            task_data[args.model] = "image none"
            result.append(task_data)
            continue

        content = [
            {"type": "image_url", "image_url": {"url": f"file://{path.resolve()}"}}
            for path in image_paths
        ]
        content.append({"type": "text", "text": prompt})

        start = time.perf_counter()
        wall_start = time.time()
        request_id = str(uuid.uuid4())
        error = None
        input_tokens = None
        output_tokens = None
        success = False
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": content}],
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            answer = response.choices[0].message.content.strip()
            usage = getattr(response, "usage", None)
            input_tokens = usage_value(usage, "prompt_tokens")
            output_tokens = usage_value(usage, "completion_tokens")
            success = True
        except Exception as exc:
            answer = "model error or image error"
            error = repr(exc)
            print(f"[{task_name}: {idx + 1}/{len(data_all)}] error: {exc}")
        finally:
            end = time.perf_counter()
            perf_records.append(
                {
                    "request_id": request_id,
                    "start_time": wall_start,
                    "end_time": time.time(),
                    "latency_s": end - start,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "success": success,
                    "error": error,
                    "model": args.model,
                    "task": task_name,
                    "sample_index": idx,
                }
            )

        print(f"[{task_name}: {idx + 1}/{len(data_all)}] {answer}")
        task_data[args.model] = answer
        result.append(task_data)

    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    perf_summary = write_perf_outputs(task_output_dir, args.model, task_name, perf_records)
    print(f"saved {output_path}")
    print(
        f"[{task_name}] perf "
        f"duration_s={fmt_metric(perf_summary['duration_s'])} "
        f"request_throughput={fmt_metric(perf_summary['request_throughput'], 4)} "
        f"mean_latency_s={fmt_metric(perf_summary['mean_latency_s'])}"
    )
    return perf_records, perf_summary


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        raise SystemExit(f"dataset dir not found: {dataset_dir}")

    tokens_per_patch, arch = read_model_hints(args.model_path, args.tokens_per_patch)
    model_name = args.model.lower().replace("-", "_")
    context_limit = args.context_limit or (23000 if "40b" in model_name else 32768)
    target_tasks = [t.strip() for t in args.target_tasks.split(",") if t.strip()]

    print("=" * 60)
    print("MMIU inference")
    print(f"api_base={args.api_base}")
    print(f"model={args.model}")
    print(f"model_path={args.model_path or '<unset>'}")
    print(f"architectures={arch}")
    print(f"dataset_dir={dataset_dir}")
    print(f"output_dir={args.output_dir}")
    print(f"tokens_per_patch={tokens_per_patch}")
    print(f"context_limit={context_limit}")
    print(f"target_tasks={target_tasks or '[all]'}")
    print("=" * 60)

    all_tasks = sorted(p.name for p in dataset_dir.iterdir() if p.is_dir() and p.name != "modelscope")
    tasks_to_run = [task for task in all_tasks if not target_tasks or task in target_tasks]
    if not tasks_to_run:
        raise SystemExit("no MMIU tasks found")

    client = OpenAI(base_url=args.api_base, api_key="EMPTY")
    all_perf_records: list[dict[str, Any]] = []
    task_perf_summaries: list[dict[str, Any]] = []
    for task_name in tasks_to_run:
        records, summary = run_task(client, task_name, args, tokens_per_patch, context_limit)
        all_perf_records.extend(records)
        if summary is not None:
            task_perf_summaries.append(summary)
    write_overall_perf(Path(args.output_dir), args.model, all_perf_records, task_perf_summaries)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
