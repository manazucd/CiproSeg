#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# =======================================================================
#  Project       : CiproSeg - LOSO U-Net Segmentation Framework
#  File          : src/ciproseg/cli/train_cli.py
#  Purpose       : Train CLI
#
#  Copyright (c) 2026 Manaz Kaleel
#  Authors       : Manaz Kaleel <manaz.ucd@gmail.com>
#
#  Created on    : 2025-11-01
#  Last modified : 2026-04-03, MK
#  Version       : 0.1.0
#  License       : Apache-2.0
#  Repository    : https://github.com/manazucd/CiproSeg
# =======================================================================
from __future__ import annotations

import sys
import argparse
from typing import Any, Dict

import yaml

from ..core.engine import Engine
from ..core.trainer import resolve_run_output_dir
from ..utils.config import load_config
from ..utils.seed import seed_everything
from ..utils.logger import setup_logging, get_logger


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CiproSeg training CLI")
    p.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    p.add_argument("--seed", type=int, default=None, help="Override trainer.seed.")
    p.add_argument("--cpu", action="store_true", help="Force CPU (overrides trainer.cpu).")
    return p


def _apply_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    cfg = {**cfg, "trainer": dict(cfg.get("trainer") or {})}
    if args.seed is not None:
        cfg["trainer"]["seed"] = int(args.seed)
    if args.cpu:
        cfg["trainer"]["cpu"] = True
    return cfg


def main() -> int:
    args = build_argparser().parse_args()

    cfg = load_config(args.config)
    cfg = _apply_overrides(cfg, args)

    run_dir = resolve_run_output_dir(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(cfg)
    logger = get_logger(cfg)

    cfg_out = run_dir / "config_resolved.yaml"
    try:
        with open(cfg_out, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        logger.info(f"[CiproSeg] Saved resolved config: {cfg_out}")
    except Exception as e:
        logger.warning(f"[CiproSeg] Could not write resolved config to {cfg_out}: {e}")

    tcfg = cfg.get("trainer") or {}
    seed = int(tcfg.get("seed", 42))
    deterministic = bool(tcfg.get("deterministic", False))
    benchmark = bool(tcfg.get("benchmark", True))

    logger.info(f"[CiproSeg] Config: {args.config}")
    logger.info(f"[CiproSeg] Run directory: {run_dir}")
    logger.info(f"[CiproSeg] seed={seed}  deterministic={deterministic}  benchmark={benchmark}")

    seed_everything(seed, deterministic=deterministic, benchmark=benchmark)

    Engine(cfg).fit()

    logger.info("[CiproSeg] Training completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
