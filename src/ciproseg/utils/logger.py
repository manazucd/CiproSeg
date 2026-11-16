#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# =======================================================================
#  Project       : CiproSeg - LOSO U-Net Segmentation Framework
#  File          : src/ciproseg/utils/logger.py
#  Purpose       : Logger
#
#  Copyright (c) 2026 Manaz Kaleel
#  Authors       : Manaz Kaleel <manaz.ucd@gmail.com>
#
#  Created on    : 2026-02-15
#  Last modified : 2026-02-17
#  Version       : 0.1.0
#  License       : Apache-2.0
#  Repository    : https://github.com/manazucd/CiproSeg
# =======================================================================
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from .dist import is_global_zero

_LOGGER: Optional[logging.Logger] = None
_LOGGER_NAME = "ciproseg"


def _lcfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return cfg.get("logger") or {}


def _ddp_env_present() -> bool:
    return "RANK" in os.environ or "WORLD_SIZE" in os.environ


def setup_logging(cfg: Dict[str, Any]) -> None:
    """
    Configures the singleton project logger with console and optional file output.
    File logging is suppressed when a DDP environment is detected but dist is not yet
    initialised, preventing multiple ranks from writing to the same log file.
    Safe to call multiple times as handlers are attached only once.
    """
    global _LOGGER

    lcfg = _lcfg(cfg)
    name = str(lcfg.get("name", _LOGGER_NAME)) or _LOGGER_NAME
    level = getattr(logging, str(lcfg.get("level", "INFO")).upper().strip(), logging.INFO)
    formatter = logging.Formatter(
        fmt=str(lcfg.get("format", "[%(levelname)s] %(message)s")),
        datefmt=str(lcfg.get("datefmt", "%Y-%m-%d %H:%M:%S")),
    )

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False 

    if getattr(logger, "_ciproseg_configured", False):
        _LOGGER = logger
        return

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if bool(lcfg.get("to_file", True)) and is_global_zero() and not _ddp_env_present():
        out_dir = Path(str((cfg.get("trainer") or {}).get("output_dir", "runs"))).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(out_dir / str(lcfg.get("filename", "train.log"))),
                                 mode="a", encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    setattr(logger, "_ciproseg_configured", True)
    _LOGGER = logger


def get_logger(cfg: Optional[Dict[str, Any]] = None) -> logging.Logger:
    """Returns the singleton logger, initialising it on first use."""
    global _LOGGER
    if _LOGGER is None:
        setup_logging(cfg or {})
    return _LOGGER
