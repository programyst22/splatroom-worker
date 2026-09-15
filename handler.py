from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
import runpod

WORK_ROOT = Path(os.environ.get("SPLATROOM_WORKDIR", "/tmp/splatroom"))
HTTP_TIMEOUT = (30, 900)


class WorkerError(RuntimeError):
    pass


def _safe_name(value: str, fallback: str = "capture.mp4") -> str:
    value = Path(value or fallback).name
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    return value[:180] or fallback


def _callback_url(data: dict[str, Any]) -> str:
    url = str(data.get("callback_url") or "").strip()
    if not url.startswith("https://"):
        raise WorkerError("callback_url must be https://")
    return url


def callback(data: dict[str, Any], payload: dict[str, Any], *, expect_json: bool = False) -> dict[str, Any] | None:
    body = {
        "job_id": data["job_id"],
        "token": data["callback_token"],
        **payload,
    }
    response = requests.post(_callback_url(data), json=body, timeout=HTTP_TIMEOUT)
    response.raise_for_status()
    if expect_json:
        return response.json()
    return None


def report(job: dict[str, Any], data: dict[str, Any], stage: str, progress: int, **extra: Any) -> None:
    progress = max(0, min(100, int(progress)))
    message = f"{stage}: {progress}%"
    try:
        runpod.serverless.progress_update(job, message)
    except Exception as exc:
        print(f"[progress] RunPod update warning: {exc}", flush=True)

    payload: dict[str, Any] = {
        "status": "running",
        "stage": stage,
        "progress": progress,
    }
    payload.update(extra)
    callback(data, payload)


def download_file(url: str, destination: Path) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with requests.get(url, stream=True, timeout=HTTP_TIMEOUT) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    handle.write(chunk)
                    total += len(chunk)
    return total


def run_command(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    printable = " ".join(str(x) for x in command)
    print(f"\n$ {printable}\n", flush=True)
    proc = subprocess.Popen(
        command,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line.rstrip(), flush=True)
    code = proc.wait()
    if code != 0:
        raise WorkerError(f"Command failed with exit code {code}: {printable}")


def newest_config(outputs: Path) -> Path:
    configs = list(outputs.rglob("config.yml"))
    if not configs:
        raise WorkerError("Training finished but no config.yml was produced")
    return max(configs, key=lambda p: p.stat().st_mtime)


def exported_ply(export_dir: Path) -> Path:
    files = list(export_dir.rglob("*.ply"))
    if not files:
        raise WorkerError("Gaussian export finished but no .ply file was produced")
    # Prefer the largest PLY if more than one is generated.
    return max(files, key=lambda p: p.stat().st_size)


def get_fresh_upload_url(data: dict[str, Any]) -> str:
    result = callback(
        data,
        {
            "action": "get_upload_url",
            "status": "running",
            "stage": "upload",
            "progress": 94,
        },
        expect_json=True,
    )
    if not result or not result.get("upload_url"):
        raise WorkerError(f"Callback did not return upload_url: {result}")
    return str(result["upload_url"])


def upload_file(url: str, source: Path) -> None:
    headers = {
        "content-type": "application/octet-stream",
        "cache-control": "max-age=3600",
        "x-upsert": "false",
    }
    with source.open("rb") as handle:
        response = requests.put(url, data=handle, headers=headers, timeout=(30, 3600))
    if response.status_code >= 400:
        raise WorkerError(f"Upload failed ({response.status_code}): {response.text[:1000]}")


def runtime_preflight() -> dict[str, Any]:
    import torch
    import gsplat

    if not torch.cuda.is_available():
        raise WorkerError("CUDA is not available in the Serverless worker")
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gsplat": getattr(gsplat, "__version__", "unknown"),
    }


def handler(job: dict[str, Any]) -> dict[str, Any]:
    started = time.time()
    data = dict(job.get("input") or {})

    required = [
        "job_id",
        "project_id",
        "input_url",
        "output_path",
        "callback_url",
        "callback_token",
    ]
    missing = [key for key in required if not data.get(key)]
    if missing:
        raise WorkerError(f"Missing required input: {', '.join(missing)}")

    runpod_job_id = str(job.get("id") or "unknown")
    job_id = str(data["job_id"])
    workdir = WORK_ROOT / job_id
    source_dir = workdir / "source"
    processed_dir = workdir / "processed"
    outputs_dir = workdir / "outputs"
    export_dir = workdir / "export"

    if workdir.exists():
        shutil.rmtree(workdir)
    for folder in (source_dir, processed_dir, outputs_dir, export_dir):
        folder.mkdir(parents=True, exist_ok=True)

    original_name = _safe_name(str(data.get("original_name") or "capture.mp4"))
    extension = Path(original_name).suffix.lower()
    if extension not in {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}:
        extension = ".mp4"
    source_file = source_dir / f"capture{extension}"

    frames_target = max(60, min(1200, int(data.get("num_frames_target") or 300)))
    iterations = max(1000, min(60000, int(data.get("max_num_iterations") or 30000)))

    metrics: dict[str, Any] = {
        "runpod_job_id": runpod_job_id,
        "frames_target": frames_target,
        "iterations": iterations,
    }

    try:
        preflight = runtime_preflight()
        metrics.update(preflight)
        report(job, data, "worker_ready", 4, metrics=metrics)

        t0 = time.time()
        report(job, data, "download", 7)
        input_bytes = download_file(str(data["input_url"]), source_file)
        metrics["input_bytes"] = input_bytes
        metrics["download_seconds"] = round(time.time() - t0, 2)
        report(job, data, "downloaded", 13, metrics=metrics)

        t0 = time.time()
        report(job, data, "camera_solve", 17)
        # Run COLMAP feature extraction/matching on CPU in headless Serverless.
        # GPU SIFT requires an OpenGL context and crashes on RunPod headless workers.
        run_command(
            [
                "ns-process-data",
                "video",
                "--data",
                str(source_file),
                "--output-dir",
                str(processed_dir),
                "--num-frames-target",
                str(frames_target),
                "--no-gpu",
            ],
            cwd=workdir,
        )
        metrics["process_seconds"] = round(time.time() - t0, 2)
        report(job, data, "camera_solved", 36, metrics=metrics)

        t0 = time.time()
        report(job, data, "training", 40)
        run_command(
            [
                "ns-train",
                "splatfacto",
                "--data",
                str(processed_dir),
                "--output-dir",
                str(outputs_dir),
                "--max-num-iterations",
                str(iterations),
                "--viewer.quit-on-train-completion",
                "True",
            ],
            cwd=workdir,
        )
        metrics["train_seconds"] = round(time.time() - t0, 2)
        config = newest_config(outputs_dir)
        metrics["config"] = str(config.relative_to(workdir))
        report(job, data, "trained", 86, metrics=metrics)

        t0 = time.time()
        report(job, data, "export", 88)
        run_command(
            [
                "ns-export",
                "gaussian-splat",
                "--load-config",
                str(config),
                "--output-dir",
                str(export_dir),
            ],
            cwd=workdir,
        )
        ply = exported_ply(export_dir)
        metrics["export_seconds"] = round(time.time() - t0, 2)
        metrics["output_bytes"] = ply.stat().st_size
        report(job, data, "exported", 93, metrics=metrics)

        # Supabase signed upload URLs are short-lived, so request one only now.
        upload_url = get_fresh_upload_url(data)
        report(job, data, "upload", 95)
        t0 = time.time()
        upload_file(upload_url, ply)
        metrics["upload_seconds"] = round(time.time() - t0, 2)
        metrics["total_seconds"] = round(time.time() - started, 2)

        callback(
            data,
            {
                "status": "succeeded",
                "stage": "published",
                "progress": 100,
                "output_path": str(data["output_path"]),
                "output_format": "ply",
                "metrics": metrics,
            },
        )
        try:
            runpod.serverless.progress_update(job, "published: 100%")
        except Exception:
            pass

        return {
            "ok": True,
            "job_id": job_id,
            "project_id": str(data["project_id"]),
            "output_path": str(data["output_path"]),
            "output_format": "ply",
            "metrics": metrics,
        }

    except Exception as exc:
        metrics["total_seconds"] = round(time.time() - started, 2)
        error_text = f"{type(exc).__name__}: {exc}"
        print(traceback.format_exc(), flush=True)
        try:
            callback(
                data,
                {
                    "status": "failed",
                    "stage": "failed",
                    "progress": 0,
                    "error_message": error_text,
                    "metrics": metrics,
                },
            )
        except Exception as callback_exc:
            print(f"Failure callback also failed: {callback_exc}", flush=True)
        raise
    finally:
        # Each Serverless job is reproducible; local files are disposable.
        try:
            shutil.rmtree(workdir, ignore_errors=True)
        except Exception:
            pass


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
