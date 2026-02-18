#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# =======================================================================
#  Project       : CiproSeg - LOSO U-Net Segmentation Framework
#  File          : src/ciproseg/cli/eval_cli.py
#  Purpose       : Evaluation CLI
#
#  Copyright (c) 2026 Manaz Kaleel
#  Authors       : Manaz Kaleel <manaz.ucd@gmail.com>
#
#  Created on    : 2025-11-01
#  Last modified : 2026-02-18, MK
#  Version       : 0.1.0
#  License       : Apache-2.0
#  Repository    : https://github.com/manazucd/CiproSeg
# =======================================================================
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict

from ..core.engine import Engine
from ..utils.config import load_config
from ..utils.logger import setup_logging, get_logger
from ..utils.seed import seed_everything


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CiproSeg evaluation CLI")
    p.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    p.add_argument(
        "--mode",
        type=str,
        default="evaluate",
        choices=["evaluate", "test"],
        help="Which engine method to run.",
    )
    p.add_argument("--seed", type=int, default=None, help="Override trainer.seed.")
    p.add_argument("--cpu", action="store_true", help="Force CPU (overrides trainer.cpu).")
    p.add_argument("--json", type=str, default=None, help="Optional path to save metrics as JSON.")
    return p


def _apply_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    # Shallow copy affected sections to avoid mutating the original config dict.
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

    setup_logging(cfg)
    logger = get_logger(cfg)

    tcfg = cfg.get("trainer") or {}
    seed_everything(
        int(tcfg.get("seed", 42)),
        deterministic=bool(tcfg.get("deterministic", False)),
        benchmark=bool(tcfg.get("benchmark", True)),
    )

    logger.info(f"[CiproSeg] Config: {args.config}  mode={args.mode}")

    engine = Engine(cfg)
    metrics = engine.test() if args.mode == "test" else engine.evaluate()
    logger.info(f"[CiproSeg] Metrics: {metrics}")

    if args.json:
        try:
            with open(args.json, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"[CiproSeg] Wrote metrics JSON: {args.json}")
        except Exception as e:
            logger.warning(f"[CiproSeg] Failed to write metrics JSON ({args.json}): {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
