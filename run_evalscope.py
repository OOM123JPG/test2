#!/usr/bin/env python3
"""Run EvalScope benchmarks against an OpenAI-compatible API."""

from __future__ import annotations

import argparse
import importlib
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import traceback
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any
from urllib.request import urlopen

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_WORK_DIR = PROJECT_DIR / "outputs" / "evalscope"
DEFAULT_PERF_OUTPUT_DIR = PROJECT_DIR / "outputs" / "evalscope_perf"
DEFAULT_DATA_DIR = PROJECT_DIR / "cache" / "data"
DEFAULT_API_URL = "http://localhost:8000/v1/chat/completions"
DEFAULT_API_KEY = "EMPTY_TOKEN"
DEFAULT_EVAL_BATCH_SIZE = 16
DEFAULT_BACKEND = "VLMEvalKit"
DEFAULT_COLLECT_PERF = False


def split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def model_dir_name(model: str) -> str:
    name = model.strip().strip("/")
    name = name.replace(os.sep, "__")
    if os.altsep:
        name = name.replace(os.altsep, "__")
    return name or "model"


def normalize_dataset_name(dataset: str) -> str:
    aliases = {
        "blink": "blink",
        "BLINK": "blink",
        "mvbench": "mvbench",
        "MVBench": "mvbench",
    }
    return aliases.get(dataset, dataset)


def parse_json_object(value: str | None, arg_name: str) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{arg_name} must be a valid JSON object: {exc}") from exc
    if not isinstance(parsed, dict):
        raise SystemExit(f"{arg_name} must be a valid JSON object")
    return parsed


def build_dataset_args(args: argparse.Namespace, datasets: list[str]) -> dict[str, dict[str, Any]]:
    dataset_args: dict[str, dict[str, Any]] = {}
    mvbench_args: dict[str, Any] = {}

    subset_list = split_csv(args.mvbench_subsets)
    if subset_list:
        mvbench_args["subset_list"] = subset_list
    mvbench_dataset_id = args.mvbench_dataset_id
    mvbench_dataset_hub = args.mvbench_dataset_hub
    if args.data_dir and not mvbench_dataset_id:
        mvbench_dataset_id = str(Path(args.data_dir).expanduser().resolve())
        mvbench_dataset_hub = mvbench_dataset_hub or "local"
    if mvbench_dataset_id:
        mvbench_args["dataset_id"] = mvbench_dataset_id
    if mvbench_dataset_hub:
        mvbench_args["dataset_hub"] = mvbench_dataset_hub
    if args.mvbench_dataset_revision:
        mvbench_args["dataset_revision"] = args.mvbench_dataset_revision

    extra_params = parse_json_object(args.mvbench_extra_params, "--mvbench-extra-params")
    if extra_params:
        mvbench_args["extra_params"] = extra_params

    for dataset in datasets:
        if dataset.lower() == "mvbench" and mvbench_args:
            dataset_args[dataset] = mvbench_args

    return dataset_args


def dataset_display_name(dataset: str) -> str:
    names = {
        "blink": "BLINK",
        "mvbench": "MVBench",
    }
    return names.get(dataset.lower(), dataset)


def vlmevalkit_dataset_name(args: argparse.Namespace, dataset: str) -> str:
    return dataset_display_name(dataset)


def select_backend(args: argparse.Namespace, dataset: str) -> str:
    if args.backend != "auto":
        return args.backend
    return DEFAULT_BACKEND


def build_native_task_kwargs(
    args: argparse.Namespace,
    dataset: str,
    dataset_work_dir: Path,
) -> dict[str, Any]:
    task_kwargs: dict[str, Any] = {
        "model": args.model,
        "api_url": args.api_url,
        "api_key": args.api_key,
        "datasets": [dataset],
        "dataset_args": build_dataset_args(args, [dataset]),
        "work_dir": str(dataset_work_dir),
        "eval_batch_size": args.eval_batch_size,
        "no_timestamp": args.no_timestamp,
        "collect_perf": args.collect_perf,
        "generation_config": {"stream": args.stream},
        "reuse": args.reuse,
    }
    if args.limit is not None:
        task_kwargs["limit"] = args.limit
    return task_kwargs


def build_vlmevalkit_task_kwargs(
    args: argparse.Namespace,
    dataset: str,
    dataset_work_dir: Path,
) -> dict[str, Any]:
    model_name = args.model.lower().replace("-", "_")
    is_internvl40b_mvbench = (
        re.search(r"internvl2?_?40b", model_name) is not None
        and dataset.lower() == "mvbench"
    )
    img_size = 448 if is_internvl40b_mvbench else -1
    mvbench_nframe = 4 if is_internvl40b_mvbench else None
    eval_config: dict[str, Any] = {
        "model": [
            {
                "type": args.model,
                "name": "CustomAPIModel",
                "api_base": args.api_url,
                "key": args.api_key,
                "temperature": 0.0,
                "timeout": args.api_timeout,
                "img_size": img_size,
                "video_llm": args.video_llm,
                "stream": args.stream,
                "max_tokens": 1024,
            }
        ],
        "data": [vlmevalkit_dataset_name(args, dataset)],
        "reuse": args.reuse,
        "reuse_aux": "infer",
        "mode": "all",
        "nproc": args.nproc,
        "judge": "exact_matching",
    }
    if mvbench_nframe is not None:
        eval_config["nframe"] = mvbench_nframe
    if args.limit is not None:
        eval_config["limit"] = args.limit

    return {
        "work_dir": str(dataset_work_dir),
        "eval_backend": "VLMEvalKit",
        "eval_batch_size": args.eval_batch_size,
        "no_timestamp": args.no_timestamp,
        "collect_perf": args.collect_perf,
        "eval_config": eval_config,
        "generation_config": {"stream": args.stream},
    }


def proxy_api_url(args: argparse.Namespace) -> str:
    return f"http://{args.perf_proxy_host}:{args.perf_proxy_port}/v1/chat/completions"


def derive_proxy_port(api_url: str) -> int:
    parsed = re.search(r":(\d+)(?:/|$)", api_url)
    if not parsed:
        return 19000
    port = int(parsed.group(1))
    derived = port + 10000
    if derived <= 65535:
        return derived
    return 19000


def start_perf_proxy(
    args: argparse.Namespace,
    *,
    target_url: str,
    model: str,
    dataset: str,
    output_dir: Path,
) -> subprocess.Popen:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = (output_dir / "proxy.log").open("a", encoding="utf-8")
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "perf_proxy.py"),
        "--listen-host",
        args.perf_proxy_host,
        "--listen-port",
        str(args.perf_proxy_port),
        "--target-url",
        target_url,
        "--output-dir",
        str(output_dir),
        "--model",
        model,
        "--dataset",
        dataset,
    ]
    if args.perf_proxy_no_include_usage:
        cmd.append("--no-include-usage")
    if args.task_type_inject:
        cmd.append("--task-type-inject")
        cmd.extend(["--task-type-data", args.task_type_data])

    process = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    process._perf_proxy_log_file = log_file  # type: ignore[attr-defined]
    wait_for_perf_proxy(proxy_api_url(args), process)
    return process


def wait_for_perf_proxy(api_url: str, process: subprocess.Popen, timeout: float = 20.0) -> None:
    models_url = models_url_from_api_url(api_url)
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        if process.poll() is not None:
            raise SystemExit(f"perf proxy exited early with code {process.returncode}")
        try:
            with urlopen(models_url, timeout=1):
                return
        except Exception as exc:
            last_error = exc
            time.sleep(0.2)
    raise SystemExit(f"perf proxy did not become ready at {models_url}: {last_error}")


def stop_perf_proxy(process: subprocess.Popen | None) -> None:
    if process is None:
        return
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
    log_file = getattr(process, "_perf_proxy_log_file", None)
    if log_file is not None:
        log_file.close()


def configure_data_dir(data_dir: str | None) -> Path | None:
    if not data_dir:
        return None

    data_path = Path(data_dir).expanduser().resolve()
    data_path.mkdir(parents=True, exist_ok=True)

    os.environ["LMUData"] = str(data_path)
    os.environ["EVALSCOPE_CACHE_DIR"] = str(data_path)
    os.environ["MODELSCOPE_CACHE"] = str(data_path / "modelscope")
    os.environ["HF_HOME"] = str(data_path / "huggingface")
    os.environ["HF_DATASETS_CACHE"] = str(data_path / "huggingface" / "datasets")
    return data_path


def ensure_mvbench_local_layout(data_path: Path | None) -> None:
    if data_path is None:
        return

    mvbench_dir = data_path / "modelscope" / "datasets" / "modelscope" / "MVBench"
    mvbench_tsv = mvbench_dir / "MVBench.tsv"
    if not mvbench_tsv.exists():
        return

    lmu_tsv = data_path / "MVBench.tsv"
    if not lmu_tsv.exists():
        try:
            lmu_tsv.symlink_to(mvbench_tsv)
            print(f"Linked local MVBench TSV for VLMEvalKit: {lmu_tsv} -> {mvbench_tsv}")
        except OSError:
            import shutil

            shutil.copy2(mvbench_tsv, lmu_tsv)
            print(f"Copied local MVBench TSV for VLMEvalKit: {lmu_tsv}")

    images_dir = data_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    mvbench_images = images_dir / "MVBench"
    candidate_image_dirs = (
        mvbench_dir / "images" / "MVBench",
        mvbench_dir / "images",
        mvbench_dir / "video",
    )
    if not mvbench_images.exists():
        for candidate in candidate_image_dirs:
            if candidate.exists():
                try:
                    mvbench_images.symlink_to(candidate, target_is_directory=True)
                    print(f"Linked local MVBench media for VLMEvalKit: {mvbench_images} -> {candidate}")
                except OSError:
                    pass
                break


def patch_mvbench_cache_md5(data_path: Path | None) -> None:
    if data_path is None:
        return

    mvbench_tsv = data_path / "modelscope" / "datasets" / "modelscope" / "MVBench" / "MVBench.tsv"
    if not mvbench_tsv.exists():
        return

    try:
        import vlmeval.dataset.mvbench as mvbench
    except Exception as exc:
        print(f"Warning: cannot patch MVBench cache MD5 before import: {exc}")
        return

    digest = hashlib.md5(mvbench_tsv.read_bytes()).hexdigest()
    patched = []
    try:
        import vlmeval.utils.dataset_config as dataset_config

        md5_dict = getattr(dataset_config, "dataset_md5_dict", None)
        if isinstance(md5_dict, dict) and md5_dict.get("MVBench") != digest:
            md5_dict["MVBench"] = digest
            patched.append("dataset_md5_dict.MVBench")
    except Exception as exc:
        print(f"Warning: cannot patch MVBench dataset_config MD5: {exc}")

    for class_name in ("MVBench", "MVBench_MP4"):
        cls = getattr(mvbench, class_name, None)
        if cls is None or getattr(cls, "MD5", None) == digest:
            continue
        cls.MD5 = digest
        patched.append(class_name)

    if patched:
        print(f"Using local MVBench.tsv cache with MD5 {digest}: {mvbench_tsv}")


def _patch_md5_value(obj: Any, dataset_name: str, digest: str) -> bool:
    patched = False
    for attr_name in ("MD5", "DATASET_MD5", "DATASET_MD5S", "dataset_md5"):
        value = getattr(obj, attr_name, None)
        if isinstance(value, str) and dataset_name.lower() in getattr(obj, "__name__", "").lower():
            if value != digest:
                setattr(obj, attr_name, digest)
                patched = True
        elif isinstance(value, dict):
            for key in list(value.keys()):
                if str(key).lower() == dataset_name.lower() and value[key] != digest:
                    value[key] = digest
                    patched = True
    return patched


def patch_blink_cache_md5(data_path: Path | None) -> None:
    if data_path is None:
        return

    blink_tsv = data_path / "BLINK.tsv"
    if not blink_tsv.exists():
        blink_tsv = data_path / "LMUData" / "BLINK.tsv"
    if not blink_tsv.exists():
        return

    digest = hashlib.md5(blink_tsv.read_bytes()).hexdigest()
    patched: list[str] = []
    module_names = (
        "vlmeval.dataset",
        "vlmeval.dataset.image_base",
        "vlmeval.dataset.image_mcq",
        "vlmeval.dataset.utils",
    )
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        if _patch_md5_value(module, "BLINK", digest):
            patched.append(module_name)
        for attr_name in dir(module):
            try:
                attr = getattr(module, attr_name)
            except Exception:
                continue
            if _patch_md5_value(attr, "BLINK", digest):
                patched.append(f"{module_name}.{attr_name}")

    if patched:
        print(f"Using local BLINK.tsv cache with MD5 {digest}: {blink_tsv}")


def install_remote_access_tracers() -> None:
    """Print a stack trace when dataset code tries to download remote resources."""
    if os.environ.get("TEXT_SVD_TRACE_REMOTE", "1") == "0":
        return

    def wrap(module_name: str, func_name: str) -> None:
        try:
            module = importlib.import_module(module_name)
            original = getattr(module, func_name, None)
        except Exception:
            return
        if original is None or getattr(original, "_text_svd_traced", False):
            return

        def traced(*args: Any, **kwargs: Any) -> Any:
            print("=" * 80)
            print(f"[TEXT_SVD_REMOTE_TRACE] {module_name}.{func_name} called")
            print(f"[TEXT_SVD_REMOTE_TRACE] args={args!r}")
            print(f"[TEXT_SVD_REMOTE_TRACE] kwargs={kwargs!r}")
            print("[TEXT_SVD_REMOTE_TRACE] stack:")
            traceback.print_stack(limit=18)
            print("=" * 80)
            return original(*args, **kwargs)

        traced._text_svd_traced = True  # type: ignore[attr-defined]
        setattr(module, func_name, traced)

    for module_name, func_name in (
        ("vlmeval.utils.dataset", "download_file"),
        ("vlmeval.smp.file", "download_file"),
        ("modelscope", "dataset_snapshot_download"),
        ("modelscope.hub.snapshot_download", "snapshot_download"),
        ("modelscope.msdatasets", "MsDataset"),
    ):
        wrap(module_name, func_name)


def print_dataset_debug(args: argparse.Namespace, data_path: Path | None) -> None:
    if os.environ.get("TEXT_SVD_DATA_DEBUG", "1") == "0":
        return
    print("=" * 80)
    print("[TEXT_SVD_DATA_DEBUG] dataset environment")
    for key in (
        "LMUData",
        "EVALSCOPE_CACHE_DIR",
        "MODELSCOPE_CACHE",
        "MODELSCOPE_OFFLINE",
        "HF_HOME",
        "HF_HUB_OFFLINE",
        "HF_DATASETS_OFFLINE",
        "TRANSFORMERS_OFFLINE",
    ):
        print(f"[TEXT_SVD_DATA_DEBUG] {key}={os.environ.get(key, '<unset>')}")
    if data_path is not None:
        paths = (
            data_path / "MVBench.tsv",
            data_path / "images" / "MVBench",
            data_path / "modelscope" / "datasets" / "modelscope" / "MVBench" / "MVBench.tsv",
            data_path / "modelscope" / "datasets" / "modelscope" / "MVBench" / "video",
        )
        for path in paths:
            print(f"[TEXT_SVD_DATA_DEBUG] exists={path.exists()} is_symlink={path.is_symlink()} path={path}")
        frame_count = 0
        frame_root = data_path / "images" / "MVBench"
        if frame_root.exists():
            frame_count = sum(1 for _ in frame_root.glob("*/*.jpg"))
        print(f"[TEXT_SVD_DATA_DEBUG] MVBench frame jpg count under images/MVBench: {frame_count}")
    try:
        import vlmeval.utils.dataset_config as dataset_config

        md5_dict = getattr(dataset_config, "dataset_md5_dict", {})
        dataset_urls = getattr(dataset_config, "dataset_URLs", {})
        print(f"[TEXT_SVD_DATA_DEBUG] VLMEvalKit MVBench url={dataset_urls.get('MVBench')}")
        print(f"[TEXT_SVD_DATA_DEBUG] VLMEvalKit MVBench md5={md5_dict.get('MVBench')}")
    except Exception as exc:
        print(f"[TEXT_SVD_DATA_DEBUG] cannot inspect VLMEvalKit dataset_config: {exc}")
    print("=" * 80)


def import_run_task():
    script_dir = str(SCRIPT_DIR)
    removed_paths: list[tuple[int, str]] = []

    for index in range(len(sys.path) - 1, -1, -1):
        path = sys.path[index] or os.getcwd()
        if Path(path).resolve() == SCRIPT_DIR:
            removed_paths.append((index, sys.path.pop(index)))

    sys.modules.pop("evalscope", None)

    try:
        try:
            return importlib.import_module("evalscope.run").run_task
        except ModuleNotFoundError as exc:
            if exc.name != "evalscope.run":
                raise
            return importlib.import_module("evalscope").run_task
    except ImportError as exc:
        raise SystemExit(
            "Cannot import the installed evalscope package. "
            f"The wrapper path is {script_dir}; please check Python/pip pairing with: python -m pip show evalscope"
        ) from exc
    finally:
        for index, path in reversed(removed_paths):
            sys.path.insert(index, path)


def filter_task_config_kwargs(task_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Keep the wrapper compatible with the locally installed EvalScope."""
    try:
        from evalscope.config import TaskConfig
    except Exception:
        return task_kwargs
    if not is_dataclass(TaskConfig):
        return task_kwargs
    supported = {field.name for field in fields(TaskConfig)}
    filtered = {key: value for key, value in task_kwargs.items() if key in supported}
    dropped = sorted(set(task_kwargs) - set(filtered))
    if dropped:
        print(f"EvalScope TaskConfig does not support keys, dropped: {dropped}")
    return filtered


def run_evalscope(args: argparse.Namespace) -> None:
    datasets = [normalize_dataset_name(dataset) for dataset in split_csv(args.datasets)]
    if not datasets:
        raise SystemExit("Please set at least one dataset with --datasets, e.g. mvbench or blink,mvbench")
    if args.collect_perf and not args.stream:
        raise SystemExit("--collect-perf requires --stream for TTFT collection; use --no-collect-perf with --no-stream")

    if not args.model:
        args.model = infer_model_name(args.api_url)
        print(f"Inferred model from API: {args.model}")

    data_path = configure_data_dir(args.data_dir)
    ensure_mvbench_local_layout(data_path)
    run_task = import_run_task()
    patch_mvbench_cache_md5(data_path)
    patch_blink_cache_md5(data_path)
    print_dataset_debug(args, data_path)
    install_remote_access_tracers()
    model_work_dir = Path(args.work_dir).expanduser().resolve() / model_dir_name(args.model)
    original_api_url = args.api_url
    if args.perf_proxy and args.perf_proxy_port is None:
        args.perf_proxy_port = derive_proxy_port(original_api_url)

    for dataset in datasets:
        dataset_work_dir = model_work_dir / dataset
        dataset_work_dir.mkdir(parents=True, exist_ok=True)
        perf_proxy_process: subprocess.Popen | None = None
        if args.perf_proxy:
            perf_output_dir = (
                Path(args.perf_output_dir).expanduser().resolve()
                / model_dir_name(args.model)
                / dataset
            )
            perf_proxy_process = start_perf_proxy(
                args,
                target_url=original_api_url,
                model=args.model,
                dataset=dataset,
                output_dir=perf_output_dir,
            )
            args.api_url = proxy_api_url(args)

        backend = select_backend(args, dataset)
        if backend == "VLMEvalKit":
            task_kwargs = build_vlmevalkit_task_kwargs(args, dataset, dataset_work_dir)
        else:
            task_kwargs = build_native_task_kwargs(args, dataset, dataset_work_dir)
        task_kwargs = filter_task_config_kwargs(task_kwargs)

        print("=" * 80)
        print("EvalScope evaluation")
        print(f"Model: {args.model}")
        print(f"API URL: {args.api_url}")
        print(f"Dataset: {dataset}")
        print(f"Backend: {backend}")
        print(f"Data dir: {data_path if data_path is not None else 'default'}")
        print(f"Output folder: {dataset_work_dir}")
        print(f"Eval batch size: {args.eval_batch_size}")
        print(f"Collect perf: {args.collect_perf}")
        print(f"Generation stream: {args.stream}")
        print(f"Reuse: {args.reuse}")
        print(f"No timestamp: {args.no_timestamp}")
        if backend == "VLMEvalKit":
            print(f"VLMEvalKit video_llm: {args.video_llm}")
            print(f"VLMEvalKit img_size: {task_kwargs['eval_config']['model'][0]['img_size']}")
            print(f"VLMEvalKit nframe: {task_kwargs['eval_config'].get('nframe', 'default')}")
            print(f"VLMEvalKit nproc: {args.nproc}")
            print(f"VLMEvalKit API timeout: {args.api_timeout}s")
        print(f"Limit: {'full' if args.limit is None else args.limit}")
        if args.perf_proxy:
            print(f"Perf proxy API URL: {args.api_url}")
            print(f"Perf output folder: {perf_output_dir}")
            print(f"Perf proxy task_type_inject: {args.task_type_inject}")
            print(f"Perf proxy task_type_data: {args.task_type_data or '<none>'}")
        print("Note: max_tokens is intentionally not set by this wrapper.")
        print("=" * 80)

        try:
            run_task(task_cfg=task_kwargs)
        finally:
            args.api_url = original_api_url
            stop_perf_proxy(perf_proxy_process)


def check_server(api_url: str) -> bool:
    try:
        models_url, model_ids = fetch_model_ids(api_url)
        print(f"vLLM server is running: {models_url}")
        print(f"Available models: {model_ids}")
        return True
    except Exception as exc:
        print(f"Warning: cannot connect to {models_url_from_api_url(api_url)}")
        print(f"Error: {exc}")
        return False


def models_url_from_api_url(api_url: str) -> str:
    return re.sub(r"/chat/completions/?$", "/models", api_url.rstrip("/"))


def fetch_model_ids(api_url: str) -> tuple[str, list[str]]:
    import json as json_lib
    import urllib.request

    models_url = models_url_from_api_url(api_url)
    with urllib.request.urlopen(models_url, timeout=5) as response:
        data = json_lib.loads(response.read().decode("utf-8"))
    model_ids = [model.get("id") for model in data.get("data", []) if model.get("id")]
    return models_url, model_ids


def infer_model_name(api_url: str) -> str:
    models_url, model_ids = fetch_model_ids(api_url)
    if not model_ids:
        raise SystemExit(f"No model id returned by {models_url}; please pass --model explicitly")
    return model_ids[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EvalScope benchmarks with an OpenAI-compatible API")
    parser.add_argument("--model", default=None, help="Model name registered by the API server; inferred from /v1/models if omitted")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="OpenAI-compatible chat completions URL")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="API key or placeholder token")
    parser.add_argument("--datasets", default="mvbench", help="Comma-separated datasets, e.g. mvbench or blink,mvbench")
    parser.add_argument("--backend", choices=["auto", "Native", "VLMEvalKit", "OpenCompass", "RAGEval", "ThirdParty"], default=DEFAULT_BACKEND, help="EvalScope backend")
    parser.add_argument("--work-dir", default=str(DEFAULT_WORK_DIR), help="Base output directory; model name is appended")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Local dataset/cache directory")
    parser.add_argument("--eval-batch-size", type=int, default=DEFAULT_EVAL_BATCH_SIZE, help="EvalScope eval_batch_size")
    parser.add_argument("--nproc", type=int, default=128, help="VLMEvalKit nproc")
    parser.add_argument("--api-timeout", type=int, default=900, help="CustomAPIModel request timeout in seconds")
    parser.add_argument("--video-llm", dest="video_llm", action="store_true", default=False, help="Let VLMEvalKit send videos as video inputs")
    parser.add_argument("--no-video-llm", dest="video_llm", action="store_false", help="Let VLMEvalKit convert videos to image/frame inputs")
    parser.add_argument("--collect-perf", dest="collect_perf", action="store_true", default=DEFAULT_COLLECT_PERF, help="Collect per-request performance metrics")
    parser.add_argument("--no-collect-perf", dest="collect_perf", action="store_false", help="Disable performance metrics collection")
    parser.add_argument("--stream", dest="stream", action="store_true", default=False, help="Set generation_config stream=true")
    parser.add_argument("--no-stream", dest="stream", action="store_false", help="Set generation_config stream=false")
    parser.add_argument("--limit", type=int, default=None, help="Number of samples; omit for full evaluation")
    parser.add_argument("--reuse", dest="reuse", action="store_true", default=True, help="Reuse existing EvalScope results/cache")
    parser.add_argument("--no-reuse", dest="reuse", action="store_false", help="Disable reuse")
    parser.add_argument("--no-timestamp", dest="no_timestamp", action="store_true", default=True, help="Do not create timestamped result dirs")
    parser.add_argument("--timestamp", dest="no_timestamp", action="store_false", help="Allow timestamped result dirs")
    parser.add_argument("--perf-proxy", action="store_true", help="Run a local proxy to record request-level performance metrics")
    parser.add_argument("--perf-proxy-host", default="127.0.0.1", help="Perf proxy listen host")
    parser.add_argument("--perf-proxy-port", type=int, default=None, help="Perf proxy listen port; defaults to API port + 10000")
    parser.add_argument("--perf-output-dir", default=str(DEFAULT_PERF_OUTPUT_DIR), help="Directory for perf proxy JSONL/summary outputs")
    parser.add_argument("--perf-proxy-no-include-usage", action="store_true", help="Do not inject stream_options.include_usage into streaming requests")
    parser.add_argument("--task-type-inject", action="store_true", help="Ask perf proxy to inject task_type into requests")
    parser.add_argument("--task-type-data", default="", help="TSV used by perf proxy task_type injection")
    parser.add_argument("--check", action="store_true", help="Only check the API server and exit")
    parser.add_argument("--mvbench-subsets", default=None, help="Comma-separated MVBench subsets")
    parser.add_argument("--mvbench-dataset-id", default=None, help="MVBench dataset repo ID or local dataset root")
    parser.add_argument("--mvbench-dataset-hub", choices=["huggingface", "modelscope", "local"], default=None)
    parser.add_argument("--mvbench-dataset-revision", default=None, help="Optional MVBench dataset revision")
    parser.add_argument("--mvbench-extra-params", default=None, help="JSON object passed as MVBench extra_params")
    args = parser.parse_args()

    if args.check:
        sys.exit(0 if check_server(args.api_url) else 1)

    run_evalscope(args)


if __name__ == "__main__":
    main()
