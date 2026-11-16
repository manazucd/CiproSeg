#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# =======================================================================
#  Project       : CiproSeg - LOSO U-Net Segmentation Framework
#  File          : src/ciproseg/utils/config.py
#  Purpose       : Config
#
#  Copyright (c) 2026 Manaz Kaleel
#  Authors       : Manaz Kaleel <manaz.ucd@gmail.com>
#
#  Created on    : 2025-11-01
#  Last modified : 2026-10-03, MK
#  Version       : 0.1.0
#  License       : Apache-2.0
#  Repository    : https://github.com/manazucd/CiproSeg
# =======================================================================
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass(frozen=True)
class ConfigPaths:
    """Resolved filesystem paths derived from config, optional convenience wrapper."""
    config_path: Path
    work_dir: Path
    output_dir: Path


def _expand_env(value: Any) -> Any:
    """Expands environment variables in string values (e.g. '${HOME}/data')."""
    return os.path.expandvars(value) if isinstance(value, str) else value


def _expand_section(cfg: Dict[str, Any], key: str) -> None:
    cfg[key] = {k: _expand_env(v) for k, v in (cfg.get(key) or {}).items()}


def load_config(path: str) -> Dict[str, Any]:
    """Loads and validates a YAML config file, returning a guaranteed non None dict."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    if not p.is_file():
        raise ValueError(f"Config path is not a file: {p}")

    try:
        with p.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in config: {p}") from e

    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        raise ValueError("Config root must be a mapping (YAML dict).")

    for section in ("trainer", "validation", "dataset", "logger"):
        cfg.setdefault(section, {})
        _expand_section(cfg, section)

    return cfg


def resolve_paths(cfg: Dict[str, Any], config_path: Optional[str] = None) -> ConfigPaths:
    """
    Resolves work_dir and output_dir from config, keeping path logic out of Engine/Trainer.
    Normalises cfg["trainer"] in place so downstream code and saved YAML stay consistent.
    """
    tcfg = cfg.get("trainer") or {}
    work_dir = Path(_expand_env(tcfg.get("work_dir", Path.cwd()))).expanduser().resolve()

    raw_out = _expand_env(tcfg.get("output_dir"))
    if not raw_out or not str(raw_out).strip():
        output_dir = work_dir / "runs"
    else:
        output_dir = Path(str(raw_out)).expanduser()
        if not output_dir.is_absolute():
            output_dir = work_dir / output_dir
        output_dir = output_dir.resolve()

    cfg_path = Path(config_path).expanduser().resolve() if config_path else Path("<memory>")

    cfg.setdefault("trainer", {})
    cfg["trainer"]["work_dir"] = str(work_dir)
    cfg["trainer"]["output_dir"] = str(output_dir)

    return ConfigPaths(config_path=cfg_path, work_dir=work_dir, output_dir=output_dir)
