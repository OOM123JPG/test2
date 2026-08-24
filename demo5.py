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
PROFILE_OFFSETS = {
    "8b": 0,
    "8b_svd_text": 10,
    "26b": 20,
    "26b_svd_text": 30,
    "40b": 40,
    "40b_svd_text": 50,
}
DEFAULT_MODEL_BY_PROFILE = {
    "8b": "internvl2_8b",
    "8b_svd_text": "internvl2_8b_svd_text_160_0703",
    "26b": "internvl2_26b",
    "26b_svd_text": "internvl2_26b_svd_text_160_0703",
    "40b": "internvl2_40b",
    "40b_svd_text": "internvl2_40b_svd_text_160_0703",
}
DEFAULT_PORT = 8000
PROBE_TIMEOUT = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one multi-image VLM question and ask for a short visible reasoning answer."
    )
    parser.add_argument("--api-base", default="", help="OpenAI API base, e.g. http://127.0.0.1:8010/v1")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--base-port", type=int, default=DEFAULT_PORT, help="Base port for profile offsets.")
    parser.add_argument("--port", type=int, default=None, help="Optional. Can be inferred from --model.")
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_OFFSETS),
        default="",
        help="Model profile used to infer --port from --base-port.",
    )
    parser.add_argument(
        "--probe-ports",
        default="",
        help="Comma-separated ports to scan when --model is given without --port.",
    )
    parser.add_argument("--list-models", action="store_true", help="Print models found on probed ports and exit.")
    parser.add_argument("--model", default="", help="Optional. Can be inferred from /v1/models or --port.")
    parser.add_argument("--source", choices=["mvbench", "mmiu"], default="mvbench")
    parser.add_argument("--task", default="multiple_image_captioning", help="MMIU task name when --source mmiu.")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--frames", type=int, default=32, help="Number of MVBench sampled frames to use. Local cache usually has 4, 8, 16, or 32.")
    parser.add_argument("--max-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--print-delay", type=float, default=0.03, help="Seconds to sleep after printing each streamed token.")
    parser.add_argument("--prefill-pause", type=float, default=1.0, help="Seconds to pause before printing streamed output.")
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument("--show-payload", action="store_true")
    parser.add_argument("--curl-only", action="store_true")
    parser.add_argument(
        "--prompt-extra",
        default="",
        help="Extra instruction appended to the default analysis prompt.",
    )
    return parser.parse_args()


def api_base_for(args: argparse.Namespace, port: int | None = None) -> str:
    if args.api_base:
        return args.api_base.rstrip("/")
    target_port = port if port is not None else args.port
    if target_port is None:
        target_port = args.base_port
    return f"http://{args.host}:{target_port}/v1"


def models_endpoint(args: argparse.Namespace, port: int | None = None) -> str:
    api_base = api_base_for(args, port)
    return f"{api_base.rstrip('/')}/models"


def fetch_served_models(args: argparse.Namespace, port: int | None = None, timeout: float = 5.0) -> list[str]:
    req = urllib.request.Request(models_endpoint(args, port), method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return [item["id"] for item in data.get("data", []) if item.get("id")]


def known_model_by_port(args: argparse.Namespace) -> dict[int, str]:
    return {
        args.base_port + offset: model
        for profile, offset in PROFILE_OFFSETS.items()
        for model in [DEFAULT_MODEL_BY_PROFILE[profile]]
    }


def port_for_profile(args: argparse.Namespace, profile: str) -> int:
    return args.base_port + PROFILE_OFFSETS[profile]


def infer_profile_from_model(model: str) -> str | None:
    normalized = model.lower().replace("-", "_")
    if "40b" in normalized:
        size = "40b"
    elif "26b" in normalized:
        size = "26b"
    elif "8b" in normalized:
        size = "8b"
    else:
        return None

    is_svd = "svd" in normalized
    return f"{size}_svd_text" if is_svd else size


def model_to_port(args: argparse.Namespace, model: str) -> int | None:
    profile = infer_profile_from_model(model)
    if profile:
        return port_for_profile(args, profile)
    for port, served_model in known_model_by_port(args).items():
        if served_model == model:
            return port
    return None


def probe_ports(args: argparse.Namespace) -> list[int]:
    if not args.probe_ports:
        return sorted(known_model_by_port(args))
    ports = []
    for item in args.probe_ports.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            ports.append(int(item))
        except ValueError as exc:
            raise SystemExit(f"Invalid port in --probe-ports: {item!r}") from exc
    return ports


def scan_served_models(args: argparse.Namespace) -> dict[int, list[str]]:
    found: dict[int, list[str]] = {}
    for port in probe_ports(args):
        try:
            models = fetch_served_models(args, port=port, timeout=PROBE_TIMEOUT)
        except Exception:
            continue
        if models:
            found[port] = models
    return found


def normalize_model_name(model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", model.lower())


def model_matches(requested: str, served: str) -> bool:
    req = normalize_model_name(requested)
    got = normalize_model_name(served)
    return req == got or req in got or got in req


def format_found_models(found: dict[int, list[str]]) -> str:
    if not found:
        return "none"
    return ", ".join(f"{port}:{models}" for port, models in sorted(found.items()))


def resolve_service(args: argparse.Namespace) -> tuple[str, str, str]:
    if args.api_base:
        models = []
        try:
            models = fetch_served_models(args)
        except Exception:
            pass
        if args.model:
            return args.model, "command line", api_base_for(args)
        if models:
            return models[0], models_endpoint(args), api_base_for(args)
        raise SystemExit("Cannot infer model from --api-base. Pass --model explicitly.")

    if args.port is not None:
        try:
            served_models = fetch_served_models(args, port=args.port)
        except Exception:
            served_models = []

        if args.model:
            if served_models and args.model not in served_models:
                print(
                    f"Warning: --model {args.model!r} is not listed by {models_endpoint(args, args.port)}; "
                    "sending it anyway."
                )
            return args.model, "command line", api_base_for(args, args.port)

        if served_models:
            mapped_model = known_model_by_port(args).get(args.port)
            if mapped_model in served_models:
                return mapped_model, models_endpoint(args, args.port), api_base_for(args, args.port)
            return served_models[0], models_endpoint(args, args.port), api_base_for(args, args.port)

        if args.port in known_model_by_port(args):
            return known_model_by_port(args)[args.port], f"fallback port map {args.port}", api_base_for(args, args.port)
        raise SystemExit(f"Cannot infer model name for port {args.port}. Pass --model explicitly.")

    if args.profile:
        port = port_for_profile(args, args.profile)
        args.port = port
        try:
            served_models = fetch_served_models(args, port=port)
        except Exception:
            served_models = []
        if args.model:
            return args.model, "command line", api_base_for(args, port)
        if served_models:
            return served_models[0], models_endpoint(args, port), api_base_for(args, port)
        return DEFAULT_MODEL_BY_PROFILE[args.profile], f"fallback profile {args.profile}", api_base_for(args, port)

    if args.model:
        found = scan_served_models(args)
        for port, models in found.items():
            if args.model in models:
                args.port = port
                return args.model, f"{models_endpoint(args, port)}", api_base_for(args, port)
        for port, models in found.items():
            for served_model in models:
                if model_matches(args.model, served_model):
                    args.port = port
                    return served_model, f"{models_endpoint(args, port)} fuzzy matched {args.model!r}", api_base_for(args, port)

        mapped_port = model_to_port(args, args.model)
        if mapped_port is not None:
            args.port = mapped_port
            try:
                served_models = fetch_served_models(args, port=mapped_port)
            except Exception:
                served_models = []
            if served_models:
                for served_model in served_models:
                    if model_matches(args.model, served_model):
                        return served_model, f"{models_endpoint(args, mapped_port)} fuzzy matched {args.model!r}", api_base_for(args, mapped_port)
                print(
                    f"Warning: inferred port {mapped_port}, but {models_endpoint(args, mapped_port)} "
                    f"lists {served_models}; sending {args.model!r} anyway."
                )
                return args.model, f"{models_endpoint(args, mapped_port)} model mismatch", api_base_for(args, mapped_port)
            source = f"fallback model profile map {mapped_port}"
            return args.model, source, api_base_for(args, mapped_port)

        if found:
            raise SystemExit(f"Model {args.model!r} was not found on probed ports. Found: {format_found_models(found)}")
        raise SystemExit(
            f"Cannot infer port for model {args.model!r}. Pass --port explicitly, "
            f"or pass --profile with --base-port."
        )

    found = scan_served_models(args)
    if found:
        port = sorted(found)[0]
        args.port = port
        models = found[port]
        mapped_model = known_model_by_port(args).get(port)
        model = mapped_model if mapped_model in models else models[0]
        return model, f"{models_endpoint(args, port)}", api_base_for(args, port)

    args.port = args.base_port
    return DEFAULT_MODEL_BY_PROFILE["8b"], f"fallback default port {args.base_port}", api_base_for(args, args.base_port)


def endpoint(args: argparse.Namespace) -> str:
    api_base = api_base_for(args)
    return f"{api_base.rstrip('/')}/chat/completions"


def endpoint_from_api_base(api_base: str) -> str:
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
    result = {
        "latency_s": total,
        "ttft_ms": (first_token_at - start) * 1000 if first_token_at is not None else None,
        "prefill_ms": (first_token_at - start) * 1000 if first_token_at is not None else None,
        "request_throughput": 1 / total if total > 0 else None,
        "finish_reason": finish_reason,
    }
    if usage:
        completion_tokens = usage.get("completion_tokens")
        prompt_tokens = usage.get("prompt_tokens")
        total_tokens = usage.get("total_tokens")
        if completion_tokens and total > 0:
            output_throughput = completion_tokens / total
        else:
            output_throughput = None
        decode_time = None
        decode_throughput = None
        if completion_tokens and first_token_at is not None:
            decode_time = max(end - first_token_at, 1e-9)
            decode_throughput = completion_tokens / decode_time
        result.update(
            {
                "prompt_tokens": prompt_tokens,
                "output_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "output_throughput": output_throughput,
                "decode_time_s": decode_time,
                "decode_throughput": decode_throughput,
            }
        )
    else:
        result["output_words"] = len(generated.split())
    return result


def stream_response(
    url: str,
    payload: dict,
    timeout: float,
    print_delay: float = 0.0,
    prefill_pause: float = 0.0,
) -> dict:
    start = time.perf_counter()
    first_token_at: float | None = None
    generated = ""
    usage: dict | None = None
    finish_reason = ""
    tokens_to_print: list[str] = []
    first_token_announced = False

    with post(url, payload, timeout) as resp:
        print(f"\nHTTP {resp.status} {resp.reason}")
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
                if not first_token_announced:
                    print("√ First token", flush=True)
                    first_token_announced = True
                generated += token
                tokens_to_print.append(token)
    end = time.perf_counter()
    print("\nModel output:")
    if prefill_pause > 0:
        time.sleep(prefill_pause)
    for token in tokens_to_print:
        print(token, end="", flush=True)
        if print_delay > 0:
            time.sleep(print_delay)
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
Evidence: 2 to 3 concise sentences based only on the frames, describing the key frames in order.
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
    if result.get("prompt_tokens") is not None:
        print(f"Prompt tokens: {result['prompt_tokens']}")
    print(f"Average latency: {result['latency_s']:.3f} s")
    if result.get("ttft_ms") is not None:
        print(f"TTFT: {result['ttft_ms']:.0f} ms")
    if result.get("prefill_ms") is not None:
        print(f"Prefill: {result['prefill_ms']:.0f} ms")
    if result.get("request_throughput") is not None:
        print(f"Request throughput: {result['request_throughput']:.3f} req/s")
    # if result.get("output_tokens") is not None:
    #     print(f"Output tokens: {result['output_tokens']}")
    if result.get("output_throughput") is not None:
        print(f"Average output throughput (e2e): {result['output_throughput']:.2f} tok/s")
        print(f"Overall output throughput (e2e): {result['output_throughput']:.2f} tok/s")
    # if result.get("decode_throughput") is not None:
    #     print(f"Decode throughput (after TTFT): {result['decode_throughput']:.2f} tok/s")
    # if result.get("finish_reason"):
    #     print(f"Finish reason: {result['finish_reason']}")
    # truncated = 1 if result.get("finish_reason") == "length" else 0
    # print(f"Truncated by max_tokens: {truncated}")


def main() -> None:
    args = parse_args()
    if args.list_models:
        found = scan_served_models(args)
        print("Probed models:")
        if found:
            for port, models in sorted(found.items()):
                print(f"  {port}: {models}")
        else:
            print("  none")
        return

    model, model_source, resolved_api_base = resolve_service(args)
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
    url = endpoint_from_api_base(resolved_api_base)

    print("=" * 72)
    print("VLM reasoning demo")
    print("=" * 72)
    print(f"Endpoint: {url}")
    print(f"Model: {model}")
    print(f"Model source: {model_source}")
    print(f"Resolved API base: {resolved_api_base}")
    print(f"Base port: {args.base_port}")
    if args.profile:
        print(f"Profile: {args.profile}")
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
    if args.source == "mvbench":
        print(f"  Requested frames: {args.frames}")
    print("Images:")
    for idx, image_url in enumerate(image_urls, 1):
        print(f"  {idx}. {image_info(image_url)}")
        print(f"     {display_image_path(image_url)}")

    print("\nPrompt:")
    print(prompt)
    if expected:
        print(f"\nReference answer: {expected}")
    print()
    # print_curl(url, payload)

    if args.show_payload:
        print("\nPayload:")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.curl_only:
        return

    try:
        print("Prefill...", flush=True)
        if args.no_stream:
            result = non_stream_response(url, payload, args.timeout)
        else:
            result = stream_response(url, payload, args.timeout, args.print_delay, args.prefill_pause)
    except urllib.error.URLError as exc:
        raise SystemExit(
            f"\nRequest failed: {exc}\n"
            f"Endpoint: {url}\n"
            f"Model: {model}\n"
            f"Model source: {model_source}\n"
            f"Hint: run with --list-models or pass the working --port explicitly."
        ) from exc

    time.sleep(0.1)
    print_summary(model, result)


if __name__ == "__main__":
    main()
