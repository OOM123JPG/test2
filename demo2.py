#!/usr/bin/env python
"""One-question, record-friendly VLM demo with a moderately detailed answer."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import shlex
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import unquote, urlparse

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "cache" / "data"
MODEL_BY_PORT = {
    8000: "internvl2_8b",
    8010: "internvl2_8b_svd_text_160_0703",
    8020: "internvl2_26b",
    8030: "internvl2_26b_svd_text_160_0703",
    8040: "internvl2_40b",
    8050: "internvl2_40b_svd_text_160_0703",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one multi-image VLM question and ask for a short visible reasoning answer."
    )
    parser.add_argument("--api-base", default="", help="OpenAI API base, e.g. http://127.0.0.1:8010/v1")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default="", help="Optional. Inferred from /v1/models or --port.")
    parser.add_argument("--source", choices=["mvbench", "mmiu"], default="mvbench")
    parser.add_argument("--task", default="multiple_image_captioning", help="MMIU task name when --source mmiu.")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--frames", type=int, default=8, help="Number of MVBench sampled frames to use.")
    parser.add_argument("--max-tokens", type=int, default=80)
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


def models_endpoint(args: argparse.Namespace) -> str:
    api_base = args.api_base or f"http://{args.host}:{args.port}/v1"
    return f"{api_base.rstrip('/')}/models"


def fetch_served_models(args: argparse.Namespace) -> list[str]:
    req = urllib.request.Request(models_endpoint(args), method="GET")
    with urllib.request.urlopen(req, timeout=5.0) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return [item["id"] for item in data.get("data", []) if item.get("id")]


def infer_model(args: argparse.Namespace) -> tuple[str, str]:
    if args.model:
        return args.model, "command line"

    try:
        served_models = fetch_served_models(args)
    except Exception:
        served_models = []

    if served_models:
        mapped_model = MODEL_BY_PORT.get(args.port)
        if mapped_model in served_models:
            return mapped_model, models_endpoint(args)
        return served_models[0], models_endpoint(args)

    if args.port in MODEL_BY_PORT:
        return MODEL_BY_PORT[args.port], f"fallback port map {args.port}"
    raise SystemExit(
        f"Cannot fetch or infer model name for port {args.port}. Pass --model explicitly, "
        f"or use one of: {', '.join(str(port) for port in sorted(MODEL_BY_PORT))}"
    )


def endpoint(args: argparse.Namespace) -> str:
    api_base = args.api_base or f"http://{args.host}:{args.port}/v1"
    return f"{api_base.rstrip('/')}/chat/completions"


def local_url(path: Path) -> str:
    return f"file://{path.expanduser().resolve()}"


def image_paths_from_row(task_dir: Path, row: dict) -> list[Path]:
    input_layer = row.get("input", row)
    if not isinstance(input_layer, dict):
        input_layer = row
    old_paths = input_layer.get("input_image_path", input_layer.get("image", []))
    if isinstance(old_paths, str):
        old_paths = [old_paths]
    return [task_dir / Path(old).name for old in old_paths]


def image_info(image_url: str) -> str:
    parsed = urlparse(image_url)
    if parsed.scheme != "file":
        return image_url

    path = Path(unquote(parsed.path))
    parts = [path.name]
    if path.exists():
        parts.append(f"{path.stat().st_size / 1024:.1f} KiB")
        try:
            from PIL import Image

            with Image.open(path) as img:
                parts.append(f"{img.width}x{img.height}")
        except Exception:
            pass
    return " | ".join(parts)


def display_image_path(image_url: str) -> str:
    parsed = urlparse(image_url)
    if parsed.scheme == "file":
        return unquote(parsed.path)
    return image_url


def build_payload(
    args: argparse.Namespace,
    model: str,
    image_urls: list[str],
    prompt: str,
) -> dict:
    content = [{"type": "image_url", "image_url": {"url": image_url}} for image_url in image_urls]
    content.append({"type": "text", "text": prompt})
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "stream": not args.no_stream,
    }
    if not args.no_stream:
        payload["stream_options"] = {"include_usage": True}
    return payload


def print_curl(url: str, payload: dict) -> None:
    compact = json.dumps(payload, ensure_ascii=False)
    print("Equivalent curl:")
    print(
        "curl -N "
        + shlex.quote(url)
        + " -H 'Content-Type: application/json' "
        + "--data "
        + shlex.quote(compact)
    )


def post(url: str, payload: dict, timeout: float):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return urllib.request.urlopen(req, timeout=timeout)


def print_metrics(
    start: float,
    end: float,
    first_token_at: float | None,
    generated: str,
    usage: dict | None,
    finish_reason: str = "",
) -> dict:
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

    result = {
        "latency_s": total,
        "ttft_ms": (first_token_at - start) * 1000 if first_token_at is not None else None,
        "finish_reason": finish_reason,
    }
    if usage:
        completion_tokens = usage.get("completion_tokens")
        prompt_tokens = usage.get("prompt_tokens")
        total_tokens = usage.get("total_tokens")
        print(f"  Prompt tokens: {prompt_tokens}")
        print(f"  Output tokens: {completion_tokens}")
        print(f"  Total tokens: {total_tokens}")
        if completion_tokens and total > 0:
            output_throughput = completion_tokens / total
            print(f"  Output throughput: {output_throughput:.2f} tok/s")
        else:
            output_throughput = None
        result.update(
            {
                "prompt_tokens": prompt_tokens,
                "output_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "output_throughput": output_throughput,
            }
        )
    else:
        print(f"  Output words: {len(generated.split())}")
    return result


def stream_response(url: str, payload: dict, timeout: float) -> dict:
    start = time.perf_counter()
    first_token_at: float | None = None
    generated = ""
    usage: dict | None = None
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
            token = (choices[0].get("delta") or {}).get("content") or ""
            if token:
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                generated += token
                print(token, end="", flush=True)
    end = time.perf_counter()
    return print_metrics(start, end, first_token_at, generated, usage, finish_reason)


def non_stream_response(url: str, payload: dict, timeout: float) -> dict:
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


def frame_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"frame-(\d+)-of-(\d+)", path.name)
    if match:
        return int(match.group(1)), path.name
    return 10**9, path.name


def load_mvbench_sample(index: int, frames: int) -> tuple[Path, dict, list[Path], str, str]:
    tsv_path = DATA_DIR / "MVBench.tsv"
    image_root = DATA_DIR / "images" / "MVBench"
    if not tsv_path.exists():
        raise SystemExit(f"MVBench TSV not found: {tsv_path}")
    if not image_root.exists():
        raise SystemExit(f"MVBench frame dir not found: {image_root}")

    csv.field_size_limit(sys.maxsize)
    with tsv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if index < 0 or index >= len(rows):
        raise SystemExit(f"--index {index} is out of range for {tsv_path}, rows={len(rows)}")

    row = rows[index]
    video = row["video"]
    frame_dir = image_root / video
    all_frames = sorted(frame_dir.glob(f"frame-*-of-{frames}.jpg"), key=frame_sort_key)
    if len(all_frames) < frames:
        available = sorted(frame_dir.glob("frame-*.jpg"), key=frame_sort_key)
        raise SystemExit(
            f"Need {frames} MVBench frames for {video}, found {len(all_frames)}. "
            f"Available examples: {[p.name for p in available[:12]]}"
        )
    selected = all_frames[:frames]
    candidates = ast.literal_eval(row["candidates"])
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    answer_text = row["answer"]
    answer_letter = next(
        (labels[i] for i, candidate in enumerate(candidates) if candidate == answer_text),
        answer_text,
    )
    prompt = build_mvbench_prompt(row, candidates, frames)
    return tsv_path, row, selected, prompt, answer_letter


def build_mvbench_prompt(row: dict, candidates: list[str], frames: int) -> str:
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    choices = "\n".join(f"{labels[i]}. {candidate}" for i, candidate in enumerate(candidates))
    return f"""You are given {frames} sampled frames from a video, ordered from early to late.

Question:
{row["question"]}

Choices:
{choices}

Answer in exactly two lines:
Evidence: one short sentence based only on the frames.
Final answer: <letter>."""


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
    if args.source == "mvbench":
        source_path, row, image_paths, prompt, expected = load_mvbench_sample(args.index, args.frames)
        task_label = row.get("task_type", "MVBench")
    else:
        source_path, image_dir, row = load_row(args.task, args.index)
        image_paths = image_paths_from_row(image_dir, row)
        if not image_paths or not all(path.exists() for path in image_paths):
            missing = [str(path) for path in image_paths if not path.exists()]
            raise SystemExit(f"Sample images are incomplete. Missing: {missing}")
        prompt = build_reasoning_prompt(row, "")
        expected = row.get("output", {}).get("output_text", "")
        task_label = args.task

    if args.prompt_extra:
        prompt += "\n" + args.prompt_extra.strip()
    image_urls = [local_url(path) for path in image_paths]
    payload = build_payload(args, model, image_urls, prompt)
    url = endpoint(args)

    print("=" * 72)
    print("VLM reasoning demo")
    print("=" * 72)
    print(f"Endpoint: {url}")
    print(f"Model: {model}")
    print(f"Model source: {model_source}")
    print(f"Streaming: {not args.no_stream}")
    print(f"Max tokens: {args.max_tokens}")
    print("\nQuestion info:")
    print(f"  Source: {args.source}")
    print(f"  Task: {task_label}")
    print(f"  Row index: {args.index}")
    if args.source == "mvbench":
        print(f"  Video: {row.get('video')}")
        if row.get("start") and row.get("end"):
            print(f"  Clip: {row.get('start')}s - {row.get('end')}s")
    print(f"  Metadata: {source_path}")
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
