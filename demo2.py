#!/usr/bin/env python
"""One-question, record-friendly VLM demo with a moderately detailed answer."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
from pathlib import Path

from demo_vllm_stream import (
    DATA_DIR,
    build_payload,
    display_image_path,
    endpoint,
    image_info,
    image_paths_from_row,
    infer_model,
    local_url,
    non_stream_response,
    print_curl,
    stream_response,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one multi-image VLM question and ask for a short visible reasoning answer."
    )
    parser.add_argument("--api-base", default="", help="OpenAI API base, e.g. http://127.0.0.1:8010/v1")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default="", help="Optional. Inferred from /v1/models or --port.")
    parser.add_argument("--task", default="multiple_image_captioning")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=220)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument("--show-payload", action="store_true")
    parser.add_argument("--curl-only", action="store_true")
    parser.add_argument(
        "--prompt-extra",
        default="",
        help="Extra instruction appended to the default analysis prompt.",
    )
    return parser.parse_args()


def load_row(task_name: str, index: int) -> tuple[Path, Path, dict]:
    json_path = DATA_DIR / "mmiu_dataset" / f"{task_name}.json"
    image_dir = DATA_DIR / "mmiu_dataset" / task_name
    if not json_path.exists():
        raise SystemExit(f"Task metadata not found: {json_path}")
    if not image_dir.exists():
        raise SystemExit(f"Task image dir not found: {image_dir}")

    rows = json.loads(json_path.read_text(encoding="utf-8"))
    if index < 0 or index >= len(rows):
        raise SystemExit(f"--index {index} is out of range for {json_path}, rows={len(rows)}")
    return json_path, image_dir, rows[index]


def build_reasoning_prompt(row: dict, extra: str) -> str:
    input_layer = row.get("input", row)
    context = input_layer.get("context", "")
    question = input_layer.get("question", "")
    options = row.get("options", input_layer.get("options", ""))
    if options and options not in context:
        context = context + "\n" + options

    prompt = f"""{context.strip()}

Question:
{question.strip()}

Please answer in 4 to 6 concise sentences:
1. Briefly describe what changes or stays consistent across the images.
2. Compare the most plausible choices using visible evidence.
3. End with exactly one line in this format: Final answer: <letter>.
Do not write a long essay."""
    if extra:
        prompt += "\n" + extra.strip()
    return prompt


def print_summary(model: str, result: dict) -> None:
    print("\n" + "=" * 72)
    print("Summary")
    print("=" * 72)
    print(f"Model: {model}")
    print("Completed samples: 1")
    print(f"Average latency: {result['latency_s']:.3f} s")
    if result.get("ttft_ms") is not None:
        print(f"Average TTFT: {result['ttft_ms']:.1f} ms")
    if result.get("output_throughput") is not None:
        print(f"Average output throughput: {result['output_throughput']:.2f} tok/s")
        print(f"Overall output throughput: {result['output_throughput']:.2f} tok/s")
    truncated = 1 if result.get("finish_reason") == "length" else 0
    print(f"Truncated by max_tokens: {truncated}")


def main() -> None:
    args = parse_args()
    model, model_source = infer_model(args)
    json_path, image_dir, row = load_row(args.task, args.index)
    image_paths = image_paths_from_row(image_dir, row)
    if not image_paths or not all(path.exists() for path in image_paths):
        missing = [str(path) for path in image_paths if not path.exists()]
        raise SystemExit(f"Sample images are incomplete. Missing: {missing}")

    image_urls = [local_url(path) for path in image_paths]
    prompt = build_reasoning_prompt(row, args.prompt_extra)
    payload = build_payload(args, model, image_urls, prompt)
    url = endpoint(args)
    expected = row.get("output", {}).get("output_text", "")

    print("=" * 72)
    print("VLM reasoning demo")
    print("=" * 72)
    print(f"Endpoint: {url}")
    print(f"Model: {model}")
    print(f"Model source: {model_source}")
    print(f"Streaming: {not args.no_stream}")
    print(f"Max tokens: {args.max_tokens}")
    print("\nQuestion info:")
    print(f"  Task: {args.task}")
    print(f"  Row index: {args.index}")
    print(f"  Source: {json_path}")
    print(f"  Image count: {len(image_urls)}")
    print("Images:")
    for idx, image_url in enumerate(image_urls, 1):
        print(f"  {idx}. {image_info(image_url)}")
        print(f"     {display_image_path(image_url)}")

    print("\nPrompt:")
    print(prompt)
    if expected:
        print(f"\nReference answer: {expected}")
    print()
    print_curl(url, payload)

    if args.show_payload:
        print("\nPayload:")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.curl_only:
        return

    try:
        if args.no_stream:
            result = non_stream_response(url, payload, args.timeout)
        else:
            result = stream_response(url, payload, args.timeout)
    except urllib.error.URLError as exc:
        raise SystemExit(f"\nRequest failed: {exc}") from exc

    time.sleep(0.1)
    print_summary(model, result)


if __name__ == "__main__":
    main()
