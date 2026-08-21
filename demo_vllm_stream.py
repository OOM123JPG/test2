#!/usr/bin/env python
"""Record-friendly smoke client for local vLLM OpenAI-compatible VLM servers."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "cache" / "data"


@dataclass
class Sample:
    image_urls: list[str]
    prompt: str
    expected: str = ""
    name: str = "sample"
    index: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream visual QA requests and print timing metrics for screen recording."
    )
    parser.add_argument("--api-base", default="", help="OpenAI API base, e.g. http://127.0.0.1:8010/v1")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default="internvl2_8b")
    parser.add_argument(
        "--preset",
        choices=["multi-image", "visual-correspondence", "art-style", "describe"],
        default="multi-image",
    )
    parser.add_argument("--num-samples", type=int, default=3, help="Number of preset samples to run.")
    parser.add_argument(
        "--task",
        default="Icon_Question_Answering_with_Spatial_Context",
        help="MMIU task name used by --preset multi-image.",
    )
    parser.add_argument("--min-images", type=int, default=3)
    parser.add_argument("--max-images", type=int, default=6)
    parser.add_argument("--pause", type=float, default=0.5, help="Seconds to pause between samples.")
    parser.add_argument("--image", action="append", default=[], help="Local image path or URL. Can be passed multiple times.")
    parser.add_argument("--prompt", default="", help="Override the preset prompt.")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--no-stream", action="store_true", help="Use a normal non-streaming request.")
    parser.add_argument("--show-payload", action="store_true")
    parser.add_argument("--curl-only", action="store_true", help="Only print the equivalent curl command.")
    return parser.parse_args()


def local_url(path: Path) -> str:
    return f"file://{path.expanduser().resolve()}"


def build_mmiu_prompt(task_name: str, task_data: dict[str, Any]) -> str:
    input_layer = task_data.get("input", task_data)
    if not isinstance(input_layer, dict):
        input_layer = task_data

    context = input_layer.get("context", "")
    question = input_layer.get("question", "")
    options_text = task_data.get("options", input_layer.get("options", ""))
    if options_text and options_text not in context:
        context = context + "\n" + options_text
    prompt = (context + "\n" + question).strip()
    return prompt + "\nPlease answer the option directly, then give one short reason."


def image_paths_from_row(task_dir: Path, row: dict[str, Any]) -> list[Path]:
    input_layer = row.get("input", row)
    if not isinstance(input_layer, dict):
        input_layer = row
    old_paths = input_layer.get("input_image_path", input_layer.get("image", []))
    if isinstance(old_paths, str):
        old_paths = [old_paths]
    return [task_dir / Path(old).name for old in old_paths]


def mmiu_samples(task_name: str, limit: int, min_images: int, max_images: int) -> list[Sample]:
    task = DATA_DIR / "mmiu_dataset" / f"{task_name}.json"
    task_dir = DATA_DIR / "mmiu_dataset" / task_name
    if not task.exists():
        raise FileNotFoundError(task)
    if not task_dir.exists():
        raise FileNotFoundError(task_dir)

    samples: list[Sample] = []
    rows = json.loads(task.read_text(encoding="utf-8"))
    for idx, row in enumerate(rows):
        paths = image_paths_from_row(task_dir, row)
        if not (min_images <= len(paths) <= max_images):
            continue
        if not paths or not all(path.exists() for path in paths):
            continue
        samples.append(
            Sample(
                image_urls=[local_url(path) for path in paths],
                prompt=build_mmiu_prompt(task_name, row),
                expected=row.get("output", {}).get("output_text", ""),
                name=task_name,
                index=idx,
            )
        )
        if len(samples) >= limit:
            break
    if not samples:
        raise FileNotFoundError(
            f"No complete samples with {min_images}-{max_images} images found in {task_dir}"
        )
    return samples


def first_existing_visual_correspondence() -> Sample:
    task = DATA_DIR / "mmiu_dataset" / "visual_correspondence_blink.json"
    task_dir = DATA_DIR / "mmiu_dataset" / "visual_correspondence_blink"
    if not task.exists():
        raise FileNotFoundError(task)

    rows = json.loads(task.read_text(encoding="utf-8"))
    for idx, row in enumerate(rows):
        paths = image_paths_from_row(task_dir, row)
        if paths and all(path.exists() for path in paths):
            prompt = row["input"].get("context") or row["input"].get("question", "")
            answer = row.get("output", {}).get("output_text", "")
            return Sample(
                image_urls=[local_url(path) for path in paths],
                prompt=prompt,
                expected=answer,
                name="visual_correspondence_blink",
                index=idx,
            )
    raise FileNotFoundError(f"No complete image pair found under {task_dir}")


def first_existing_art_style() -> Sample:
    blink_dir = DATA_DIR / "images" / "BLINK"
    candidates = [
        [
            blink_dir / "val_Art_Style_2_1.jpg",
            blink_dir / "val_Art_Style_2_2.jpg",
            blink_dir / "val_Art_Style_2_3.jpg",
        ],
        [
            blink_dir / "val_Art_Style_8_1.jpg",
            blink_dir / "val_Art_Style_8_2.jpg",
            blink_dir / "val_Art_Style_8_3.jpg",
        ],
    ]
    for paths in candidates:
        existing = [path for path in paths if path.exists()]
        if len(existing) >= 2:
            prompt = (
                "Use the first image as the reference painting. Which later image has the more similar "
                "art style? Answer with a short explanation."
            )
            return Sample(
                image_urls=[local_url(path) for path in existing],
                prompt=prompt,
                name="art_style",
            )
    raise FileNotFoundError(f"No art-style images found under {blink_dir}")


def describe_preset() -> Sample:
    image = DATA_DIR / "images" / "BLINK" / "val_Functional_Correspondence_1_2.jpg"
    if not image.exists():
        image = next((DATA_DIR / "images" / "BLINK").glob("*.jpg"))
    return Sample(
        image_urls=[local_url(image)],
        prompt="Describe the image in one concise paragraph.",
        name="describe",
    )


def build_samples(args: argparse.Namespace) -> list[Sample]:
    if args.image:
        urls = []
        for item in args.image:
            if item.startswith(("http://", "https://", "file://")):
                urls.append(item)
            else:
                urls.append(local_url(Path(item)))
        return [
            Sample(
                image_urls=urls,
                prompt=args.prompt or "Answer the question based on the image(s).",
                name="custom",
            )
        ]

    if args.preset == "multi-image":
        samples = mmiu_samples(args.task, args.num_samples, args.min_images, args.max_images)
    elif args.preset == "visual-correspondence":
        samples = [first_existing_visual_correspondence()]
    elif args.preset == "art-style":
        samples = [first_existing_art_style()]
    else:
        samples = [describe_preset()]

    if args.prompt:
        for sample in samples:
            sample.prompt = args.prompt
    return samples


def build_payload(args: argparse.Namespace, image_urls: list[str], prompt: str) -> dict[str, Any]:
    content: list[dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": image_url}} for image_url in image_urls
    ]
    content.append({"type": "text", "text": prompt})
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": [{"role": "user", "content": content}],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "stream": not args.no_stream,
    }
    if not args.no_stream:
        payload["stream_options"] = {"include_usage": True}
    return payload


def endpoint(args: argparse.Namespace) -> str:
    api_base = args.api_base or f"http://{args.host}:{args.port}/v1"
    return f"{api_base.rstrip('/')}/chat/completions"


def print_curl(url: str, payload: dict[str, Any]) -> None:
    compact = json.dumps(payload, ensure_ascii=False)
    print("Equivalent curl:")
    print(
        "curl -N "
        + shlex.quote(url)
        + " -H 'Content-Type: application/json' "
        + "--data "
        + shlex.quote(compact)
    )


def post(url: str, payload: dict[str, Any], timeout: float):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return urllib.request.urlopen(req, timeout=timeout)


def stream_response(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    start = time.perf_counter()
    first_token_at: float | None = None
    generated = ""
    usage: dict[str, Any] | None = None
    finish_reason = ""

    with post(url, payload, timeout) as resp:
        print(f"\nHTTP {resp.status} {resp.reason}")
        print("\nModel output:")
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            data = line.removeprefix("data:").strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage"):
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            if not choices:
                continue
            if choices[0].get("finish_reason"):
                finish_reason = choices[0]["finish_reason"]
            delta = choices[0].get("delta") or {}
            token = delta.get("content") or ""
            if token:
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                generated += token
                print(token, end="", flush=True)
    end = time.perf_counter()
    return print_metrics(start, end, first_token_at, generated, usage, finish_reason)


def non_stream_response(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    start = time.perf_counter()
    with post(url, payload, timeout) as resp:
        body = resp.read().decode("utf-8")
    end = time.perf_counter()
    data = json.loads(body)
    choice = data["choices"][0]
    text = choice["message"]["content"]
    print("\nModel output:")
    print(text)
    return print_metrics(start, end, None, text, data.get("usage"), choice.get("finish_reason", ""))


def print_metrics(
    start: float,
    end: float,
    first_token_at: float | None,
    generated: str,
    usage: dict[str, Any] | None,
    finish_reason: str = "",
) -> dict[str, Any]:
    total = end - start
    print("\n\nTiming:")
    if first_token_at is not None:
        print(f"  TTFT: {(first_token_at - start) * 1000:.1f} ms")
    else:
        print("  TTFT: n/a")
    print(f"  Total latency: {total:.3f} s")
    if finish_reason:
        print(f"  Finish reason: {finish_reason}")
        if finish_reason == "length":
            print("  Note: output hit max_tokens; increase --max-tokens for a complete answer.")
    if usage:
        completion_tokens = usage.get("completion_tokens")
        prompt_tokens = usage.get("prompt_tokens")
        total_tokens = usage.get("total_tokens")
        print(f"  Prompt tokens: {prompt_tokens}")
        print(f"  Output tokens: {completion_tokens}")
        print(f"  Total tokens: {total_tokens}")
        if completion_tokens and total > 0:
            print(f"  Output throughput: {completion_tokens / total:.2f} tok/s")
        return {
            "latency_s": total,
            "ttft_ms": (first_token_at - start) * 1000 if first_token_at is not None else None,
            "prompt_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "finish_reason": finish_reason,
        }
    else:
        approx_tokens = len(generated.split())
        print(f"  Output words: {approx_tokens}")
        return {
            "latency_s": total,
            "ttft_ms": (first_token_at - start) * 1000 if first_token_at is not None else None,
            "output_words": approx_tokens,
            "finish_reason": finish_reason,
        }


def main() -> None:
    args = parse_args()
    samples = build_samples(args)
    url = endpoint(args)

    print("=" * 72)
    print("VLM service smoke test")
    print("=" * 72)
    print(f"Endpoint: {url}")
    print(f"Model: {args.model}")
    print(f"Streaming: {not args.no_stream}")
    print(f"Samples: {len(samples)}")
    print(f"Max tokens: {args.max_tokens}")

    results: list[dict[str, Any]] = []
    try:
        for sample_no, sample in enumerate(samples, 1):
            payload = build_payload(args, sample.image_urls, sample.prompt)
            print("\n" + "-" * 72)
            print(f"Sample {sample_no}/{len(samples)}: {sample.name}[{sample.index}]")
            print("-" * 72)
            print("Images:")
            for idx, image_url in enumerate(sample.image_urls, 1):
                print(f"  {idx}. {image_url}")
            print("\nQuestion:")
            print(sample.prompt)
            if sample.expected:
                print(f"\nReference answer: {sample.expected}")
            print()
            print_curl(url, payload)

            if args.show_payload:
                print("\nPayload:")
                print(json.dumps(payload, ensure_ascii=False, indent=2))

            if not args.curl_only:
                if args.no_stream:
                    result = non_stream_response(url, payload, args.timeout)
                else:
                    result = stream_response(url, payload, args.timeout)
                result.update({"sample": sample_no, "name": sample.name, "index": sample.index})
                results.append(result)
                if sample_no < len(samples) and args.pause > 0:
                    time.sleep(args.pause)
    except urllib.error.URLError as exc:
        raise SystemExit(f"\nRequest failed: {exc}") from exc

    if results:
        avg_latency = sum(item["latency_s"] for item in results) / len(results)
        ttfts = [item["ttft_ms"] for item in results if item.get("ttft_ms") is not None]
        avg_ttft = sum(ttfts) / len(ttfts) if ttfts else None
        print("\n" + "=" * 72)
        print("Summary")
        print("=" * 72)
        print(f"Completed samples: {len(results)}")
        print(f"Average latency: {avg_latency:.3f} s")
        if avg_ttft is not None:
            print(f"Average TTFT: {avg_ttft:.1f} ms")
        truncated = sum(1 for item in results if item.get("finish_reason") == "length")
        print(f"Truncated by max_tokens: {truncated}")


if __name__ == "__main__":
    main()
