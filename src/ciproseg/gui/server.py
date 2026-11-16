#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# =======================================================================
#  Project       : CiproSeg - LOSO U-Net Segmentation Framework
#  File          : src/ciproseg/gui/server.py
#  Purpose       : FastAPI backend for the CiproSeg web GUI
#
#  Copyright (c) 2026 Manaz Kaleel
#  Authors       : Manaz Kaleel <manaz.ucd@gmail.com>
#
#  Created on    : 2026-04-11
#  Last modified : 2026-11-16, MK
#  Version       : 0.1.0
#  License       : Apache-2.0
#  Repository    : https://github.com/manazucd/CiproSeg
# =======================================================================
"""
FastAPI server powering the CiproSeg localhost GUI.

Wraps the existing CLI tools (ciproseg-train, ciproseg-eval, ciproseg-infer)
as background subprocesses and provides REST endpoints for the browser SPA.
"""
from __future__ import annotations

import json
import logging
import os
import platform
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logger = logging.getLogger("ciproseg.gui")

app = FastAPI(title="CiproSeg GUI", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_STATIC_DIR = Path(__file__).parent / "static"

_job_lock = threading.Lock()
_active_job: Dict[str, Any] = {
    "process": None,       
    "kind": None,          
    "log_path": None,      
    "started_at": None,    
    "config": None,        
    "status": "idle",      
}


def _resolve_runs_root(root: Optional[str]) -> Path:
    """Resolve the runs directory from a client-supplied path (from the config form)."""
    if root:
        p = Path(root).expanduser()
        return p if p.is_absolute() else Path.cwd() / p
    return Path.cwd() / "runs"


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index = _STATIC_DIR / "index.html"
    if not index.exists():
        raise HTTPException(500, "Frontend not found. Ensure gui/static/index.html exists.")
    return HTMLResponse(index.read_text(encoding="utf-8"))


@app.get("/api/env")
async def get_env():
    info: Dict[str, str] = {
        "python": platform.python_version(),
        "os": f"{platform.system()} {platform.release()}",
        "ciproseg": "0.1.0",
    }
    try:
        import torch
        info["pytorch"] = torch.__version__
        info["cuda_available"] = str(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda or "N/A"
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = str(torch.cuda.device_count())
        else:
            info["gpu"] = "N/A (CPU only)"
    except ImportError:
        info["pytorch"] = "not installed"
    return info


def _scan_runs(root: Path) -> List[Dict[str, Any]]:
    """Walk runs/<model>/<timestamp>/ and collect metadata."""
    results: List[Dict[str, Any]] = []
    if not root.is_dir():
        return results

    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith("."):
            continue
        for ts_dir in sorted(model_dir.iterdir(), reverse=True):
            if not ts_dir.is_dir():
                continue
            entry: Dict[str, Any] = {
                "model": model_dir.name,
                "timestamp": ts_dir.name,
                "path": str(ts_dir),
                "folds": [],
            }

            summary_json = ts_dir / "run_summary.json"
            if summary_json.exists():
                try:
                    entry["summary"] = json.loads(summary_json.read_text("utf-8"))
                except Exception:
                    pass

            cfg_yaml = ts_dir / "config_resolved.yaml"
            if cfg_yaml.exists():
                try:
                    entry["config"] = yaml.safe_load(cfg_yaml.read_text("utf-8"))
                except Exception:
                    pass

            for fold_dir in sorted(ts_dir.iterdir()):
                if not fold_dir.is_dir() or not fold_dir.name.startswith("fold_"):
                    continue
                fold_info: Dict[str, Any] = {"name": fold_dir.name, "path": str(fold_dir)}
                best_json = fold_dir / "best.json"
                if best_json.exists():
                    try:
                        fold_info["best"] = json.loads(best_json.read_text("utf-8"))
                    except Exception:
                        pass
                last_json = fold_dir / "last.json"
                if last_json.exists():
                    try:
                        fold_info["last"] = json.loads(last_json.read_text("utf-8"))
                    except Exception:
                        pass
                fold_info["has_loss_plot"] = (fold_dir / "loss.png").exists()
                fold_info["has_dice_plot"] = (fold_dir / "dice.png").exists()
                entry["folds"].append(fold_info)

            results.append(entry)
    return results


@app.get("/api/runs")
async def list_runs(root: Optional[str] = Query(None)):
    return _scan_runs(_resolve_runs_root(root))


@app.get("/api/runs/{model}/{timestamp}")
async def get_run_detail(model: str, timestamp: str, root: Optional[str] = Query(None)):
    run_dir = _resolve_runs_root(root) / model / timestamp
    if not run_dir.is_dir():
        raise HTTPException(404, f"Run not found: {model}/{timestamp}")

    detail: Dict[str, Any] = {"model": model, "timestamp": timestamp, "path": str(run_dir)}

    summary_json = run_dir / "run_summary.json"
    if summary_json.exists():
        try:
            detail["summary"] = json.loads(summary_json.read_text("utf-8"))
        except Exception:
            pass

    cfg_yaml = run_dir / "config_resolved.yaml"
    if cfg_yaml.exists():
        try:
            detail["config"] = yaml.safe_load(cfg_yaml.read_text("utf-8"))
        except Exception:
            pass

    folds = []
    for fold_dir in sorted(run_dir.iterdir()):
        if not fold_dir.is_dir() or not fold_dir.name.startswith("fold_"):
            continue
        fi: Dict[str, Any] = {"name": fold_dir.name, "path": str(fold_dir)}
        for sidecar in ("best.json", "last.json"):
            p = fold_dir / sidecar
            if p.exists():
                try:
                    fi[sidecar.replace(".json", "")] = json.loads(p.read_text("utf-8"))
                except Exception:
                    pass
        fi["has_loss_plot"] = (fold_dir / "loss.png").exists()
        fi["has_dice_plot"] = (fold_dir / "dice.png").exists()
        folds.append(fi)
    detail["folds"] = folds
    return detail


_SAFE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".svg", ".json", ".yaml", ".yml"}


@app.get("/api/file")
async def serve_file(path: str = Query(...)):
    """Serve a file from disk. Restricted to image / config extensions."""
    p = Path(path).resolve()
    if not p.exists():
        raise HTTPException(404, f"Not found: {path}")
    if p.suffix.lower() not in _SAFE_EXTENSIONS:
        raise HTTPException(403, f"Extension not allowed: {p.suffix}")
    return FileResponse(str(p))


_INFER_SUFFIXES = ("_prob.png", "_binary.png", "_pred_overlay.png", "_gt_overlay.png", "_diff.png")


def _group_inference_outputs(output_dir: str) -> List[Dict[str, Any]]:
    """Group inference output files by their input stem."""
    d = Path(output_dir)
    if not d.is_dir():
        return []

    stems: Dict[str, Dict[str, str]] = {}
    for f in sorted(d.iterdir()):
        if not f.is_file():
            continue
        name = f.name
        for suffix in _INFER_SUFFIXES:
            if name.endswith(suffix):
                stem = name[: -len(suffix)]
                stems.setdefault(stem, {})
                key = suffix.strip("_").replace(".png", "")
                stems[stem][key] = str(f)
                break

    result = []
    for stem in sorted(stems):
        entry = {"stem": stem, "outputs": stems[stem]}
        result.append(entry)
    return result


@app.get("/api/infer/images")
async def list_infer_images(dir: str = Query(...)):
    resolved = str(Path(dir).resolve())
    images = _group_inference_outputs(resolved)
    if not images:
        p = Path(resolved)
        exists = p.exists()
        is_dir = p.is_dir()
        contents = []
        if is_dir:
            contents = [f.name for f in sorted(p.iterdir())[:20]]
        return JSONResponse({
            "images": [],
            "debug": {
                "searched_dir": resolved,
                "exists": exists,
                "is_dir": is_dir,
                "sample_contents": contents,
            },
        })
    return images


@app.get("/api/train/plots")
async def get_training_plots(run_dir: str = Query(...)):
    """Return paths to all loss.png and dice.png files in a run directory."""
    d = Path(run_dir).resolve()
    if not d.is_dir():
        return {"plots": []}
    plots = []
    for fold_dir in sorted(d.iterdir()):
        if not fold_dir.is_dir() or not fold_dir.name.startswith("fold_"):
            continue
        entry: Dict[str, Any] = {"fold": fold_dir.name}
        loss_p = fold_dir / "loss.png"
        dice_p = fold_dir / "dice.png"
        if loss_p.exists():
            entry["loss"] = str(loss_p)
            entry["loss_mtime"] = loss_p.stat().st_mtime
        if dice_p.exists():
            entry["dice"] = str(dice_p)
            entry["dice_mtime"] = dice_p.stat().st_mtime
        if "loss" in entry or "dice" in entry:
            plots.append(entry)
    return {"plots": plots}

class TrainRequest(BaseModel):
    config_path: Optional[str] = None
    config_dict: Optional[Dict[str, Any]] = None
    seed: Optional[int] = None
    cpu: bool = False


class EvalRequest(BaseModel):
    ckpt: str
    dataset_root: Optional[str] = None
    patch: Optional[int] = None
    stride: Optional[int] = None
    thr: float = 0.5
    cpu: bool = False


class InferRequest(BaseModel):
    ckpt: str
    input_dir: str
    output_dir: Optional[str] = None
    masks_dir: Optional[str] = None
    threshold: float = 0.5
    patch: int = 256
    stride: Optional[int] = None
    cpu: bool = False


def _launch_subprocess(cmd: List[str], kind: str, cfg: Optional[Dict] = None) -> Dict[str, Any]:
    """Launch a CLI command as a background subprocess."""
    with _job_lock:
        proc = _active_job.get("process")
        if proc is not None and proc.poll() is None:
            raise HTTPException(409, "A job is already running. Stop it first.")

        log_fd, log_path = tempfile.mkstemp(prefix=f"ciproseg_{kind}_", suffix=".log")
        log_file = os.fdopen(log_fd, "w")

        child_env = {**os.environ, "MPLBACKEND": "Agg"}

        try:
            process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                env=child_env,
                preexec_fn=os.setsid if sys.platform != "win32" else None,
            )
        except FileNotFoundError:
            log_file.close()
            raise HTTPException(500, f"Command not found: {cmd[0]}. Is CiproSeg installed?")

        _active_job.update({
            "process": process,
            "log_file": log_file,
            "kind": kind,
            "log_path": log_path,
            "started_at": datetime.now().isoformat(),
            "config": cfg,
            "status": "running",
        })

    def _monitor():
        process.wait()
        log_file.close()
        with _job_lock:
            _active_job["status"] = "done" if process.returncode == 0 else "failed"

    threading.Thread(target=_monitor, daemon=True).start()

    return {"status": "started", "kind": kind, "log_path": log_path}


@app.post("/api/train")
async def start_training(req: TrainRequest):
    if req.config_dict and not req.config_path:
        fd, tmp = tempfile.mkstemp(prefix="ciproseg_cfg_", suffix=".yaml")
        with os.fdopen(fd, "w") as f:
            yaml.safe_dump(req.config_dict, f, sort_keys=False)
        config_path = tmp
    elif req.config_path:
        config_path = req.config_path
    else:
        raise HTTPException(400, "Provide config_path or config_dict.")

    cmd = [sys.executable, "-m", "ciproseg.cli.train_cli", "--config", config_path]
    if req.seed is not None:
        cmd += ["--seed", str(req.seed)]
    if req.cpu:
        cmd.append("--cpu")

    return _launch_subprocess(cmd, "train", cfg=req.config_dict)


@app.post("/api/eval")
async def start_eval(req: EvalRequest):
    cmd = [sys.executable, "-m", "ciproseg.cli.eval_cli", "--ckpt", req.ckpt, "--thr", str(req.thr)]
    if req.dataset_root:
        cmd += ["--dataset-root", req.dataset_root]
    if req.patch is not None:
        cmd += ["--patch", str(req.patch)]
    if req.stride is not None:
        cmd += ["--stride", str(req.stride)]
    if req.cpu:
        cmd.append("--cpu")

    json_fd, json_path = tempfile.mkstemp(prefix="ciproseg_eval_", suffix=".json")
    os.close(json_fd)
    cmd += ["--json", json_path]

    result = _launch_subprocess(cmd, "eval")
    result["json_path"] = json_path
    return result


@app.post("/api/infer")
async def start_infer(req: InferRequest):
    if req.output_dir:
        resolved_output = str(Path(req.output_dir).resolve())
    else:
        p = Path(req.input_dir).resolve()
        base = p.parents[1] if len(p.parents) > 1 else p.parent
        resolved_output = str(base / f"{p.name}_predictions")

    cmd = [
        sys.executable, "-m", "ciproseg.cli.infer_cli",
        "--ckpt", req.ckpt,
        "--input", req.input_dir,
        "--output", resolved_output,
        "--threshold", str(req.threshold),
        "--patch", str(req.patch),
    ]
    if req.masks_dir:
        cmd += ["--masks", req.masks_dir]
    if req.stride is not None:
        cmd += ["--stride", str(req.stride)]
    if req.cpu:
        cmd.append("--cpu")
    cmd.append("--verbose")

    result = _launch_subprocess(cmd, "infer")
    result["output_dir"] = resolved_output
    return result


@app.get("/api/job/status")
async def job_status():
    with _job_lock:
        proc = _active_job.get("process")
        info = {
            "kind": _active_job["kind"],
            "status": _active_job["status"],
            "started_at": _active_job["started_at"],
            "log_path": _active_job["log_path"],
        }
        if proc is not None:
            info["pid"] = proc.pid
            info["returncode"] = proc.returncode
    return info


@app.get("/api/job/log")
async def job_log(offset: int = Query(0)):
    """Return log content from offset (byte position). Enables incremental polling."""
    with _job_lock:
        log_path = _active_job.get("log_path")
        status = _active_job["status"]
    if not log_path or not os.path.exists(log_path):
        return {"lines": [], "offset": 0, "status": status}

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        f.seek(offset)
        content = f.read()
        new_offset = f.tell()
    lines = content.splitlines() if content else []
    return {"lines": lines, "offset": new_offset, "status": status}


@app.post("/api/job/stop")
async def stop_job():
    with _job_lock:
        proc = _active_job.get("process")
        if proc is None or proc.poll() is not None:
            return {"status": "no_active_job"}
        try:
            if sys.platform != "win32":
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            else:
                proc.terminate()
            _active_job["status"] = "stopped"
        except Exception as e:
            raise HTTPException(500, f"Failed to stop: {e}")
    return {"status": "stopped"}

_CONFIG_SEARCH_PATHS = [
    Path("configs/experiment.yaml"),
    Path("configs/config_default.yaml"),
    Path("config/experiment.yaml"),
    Path("experiment.yaml"),
]

_FALLBACK_CONFIG: Dict[str, Any] = {
    "dataset": {
        "name": "ciproseg_ecoli",
        "root": "/path/to/your/Data/original_size",
        "train_frac": 0.8,
        "patch": 256,
        "stride": 128,
    },
    "model": {
        "name": "unet4",
        "in_channels": 1,
        "out_channels": 1,
        "base_channels": 64,
    },
    "trainer": {
        "epochs": 500,
        "batch_size": 8,
        "num_workers": 4,
        "seed": 1234,
        "target_patches_per_epoch": 2000,
        "deterministic": False,
        "benchmark": True,
        "cpu": False,
        "output_dir": "runs",
    },
    "validation": {
        "checkpoint_threshold": 0.5,
        "threshold_sweep": True,
        "sweeping_thresholds": [
            0.30,
            0.35,
            0.40,
            0.45,
            0.50,
            0.55,
            0.60,
        ],
    },
    "optim": {"name": "adamw", "lr": 3e-4, "weight_decay": 1e-2},
    "sched": {"name": "cosine", "tmax": 500},
    "logger": {"name": "ciproseg", "level": "INFO"},
}


def _find_default_config() -> Dict[str, Any]:
    """
    Try to load configs/experiment.yaml (or similar) from the current
    working directory. Falls back to hardcoded defaults if not found.
    Returns a tuple of (config_dict, source_path_or_None).
    """
    for candidate in _CONFIG_SEARCH_PATHS:
        p = Path.cwd() / candidate
        if p.is_file():
            try:
                cfg = yaml.safe_load(p.read_text("utf-8"))
                if isinstance(cfg, dict):
                    logger.info(f"Loaded default config from: {p}")
                    cfg["_source"] = str(p)
                    return cfg
            except Exception:
                continue
    return {**_FALLBACK_CONFIG, "_source": None}


@app.get("/api/config/default")
async def default_config():
    return _find_default_config()


@app.post("/api/config/validate")
async def validate_config(cfg: Dict[str, Any]):
    """Basic sanity checks on a config dict before launching training."""
    errors: List[str] = []
    dcfg = cfg.get("dataset") or {}
    if not dcfg.get("root"):
        errors.append("dataset.root is required.")
    elif not Path(str(dcfg["root"])).expanduser().is_dir():
        errors.append(f"dataset.root directory not found: {dcfg['root']}")

    mcfg = cfg.get("model") or {}
    if mcfg.get("name") not in ("unet4", "unet2"):
        errors.append(f"model.name must be 'unet4' or 'unet2', got '{mcfg.get('name')}'")

    tcfg = cfg.get("trainer") or {}
    epochs = tcfg.get("epochs", 0)
    if not isinstance(epochs, int) or epochs < 1:
        errors.append("trainer.epochs must be a positive integer.")

    return {"valid": len(errors) == 0, "errors": errors}


def find_free_port(host: str, start: int, max_attempts: int = 20) -> int:
    """Find the first free port starting from `start`, trying up to `max_attempts` ports."""
    import socket
    for offset in range(max_attempts):
        port = start + offset
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((host, port))
                return port
        except OSError:
            continue
    raise RuntimeError(
        f"No free port found in range {start}-{start + max_attempts - 1}. "
        "Close other instances or specify a different --port."
    )


def run_server(host: str = "127.0.0.1", port: int = 7860, open_browser: bool = True) -> None:
    """Start the Uvicorn server. Called by the ciproseg-gui CLI."""
    import uvicorn

    actual_port = find_free_port(host, port)
    if actual_port != port:
        logger.info(f"Port {port} in use, using {actual_port} instead.")

    if open_browser:
        import webbrowser
        threading.Timer(1.5, lambda: webbrowser.open(f"http://{host}:{actual_port}")).start()

    print(f"\n  CiproSeg GUI: http://{host}:{actual_port}\n")
    uvicorn.run(app, host=host, port=actual_port, log_level="info")
