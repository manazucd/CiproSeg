#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# =======================================================================
#  Project       : CiproSeg - LOSO U-Net Segmentation Framework
#  File          : src/ciproseg/utils/checkpoint.py
#  Purpose       : Checkpointing
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

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .dist import is_global_zero


class Checkpointer:
    """
    Handles two levels of persistence for LOSO training:

    Per fold: best checkpoint by validation metric:
        {output_dir}/fold_XX/best.pt  (+  best.json sidecar)

    Run level: summary written after all folds complete:
        {output_dir}/run_summary.pt  (+  run_summary.json)

    Mid fold resume is intentionally not supported.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.dir = (cfg.get("trainer") or {}).get("output_dir", "runs")
        os.makedirs(self.dir, exist_ok=True)
        self.run_path = os.path.join(self.dir, "run_summary.pt")
        self.run_json_path = os.path.join(self.dir, "run_summary.json")

    def maybe_load(self, trainer: Any) -> Optional[Dict[str, Any]]:
        if not os.path.exists(self.run_path):
            return None
        try:
            return torch.load(self.run_path, map_location="cpu")
        except Exception:
            return None

    def save_run_summary(self, trainer: Any, metric_value: float) -> None:
        if not is_global_zero():
            return

        state: Dict[str, Any] = {
            "best_metric": float(metric_value),
            "last_metrics": getattr(trainer, "last_metrics", {}),
            "fold_results": getattr(trainer, "fold_results", []),
            "cfg": self.cfg,
        }

        tmp_fd, tmp_path = tempfile.mkstemp(dir=self.dir, prefix="tmp_run_", suffix=".pt")
        os.close(tmp_fd)
        torch.save(state, tmp_path)
        os.replace(tmp_path, self.run_path)

        try:
            with open(self.run_json_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, default=str)
        except Exception:
            pass

    def _fold_dir(self, fold_id: int) -> str:
        d = os.path.join(self.dir, f"fold_{fold_id:02d}")
        os.makedirs(d, exist_ok=True)
        return d
    
    def save_fold_last(
        self,
        fold_id: int,
        fold_name: str,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler_state: Optional[Dict[str, Any]],
        best_val_metric: float,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        if not is_global_zero():
            return ""

        dpath = self.fold_dir(fold_id, fold_name)   # .../fold_XX/{fold_name}
        path = dpath / "last.pt"
        json_path = dpath / "last.json"

        state: Dict[str, Any] = {
            "fold_id": int(fold_id),
            "fold_name": str(fold_name),
            "epoch": int(epoch),
            "best_val_metric": float(best_val_metric),
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
            "sched": scheduler_state,
            "cfg": self.cfg,
        }
        if extra:
            state.update(extra)

        tmp_fd, tmp_path = tempfile.mkstemp(dir=str(dpath), prefix="tmp_last_", suffix=".pt")
        os.close(tmp_fd)
        torch.save(state, tmp_path)
        os.replace(tmp_path, str(path))

        try:
            meta = {
                "fold_id": int(fold_id),
                "fold_name": str(fold_name),
                "epoch": int(epoch),
                "best_val_metric": float(best_val_metric),
                "extra": extra or {},
            }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, default=str)
        except Exception:
            pass

        return str(path)


    def save_fold_best(
        self,
        fold_id: int,
        fold_name: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler_state: Optional[Dict[str, Any]],
        best_val_metric: float,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save the best checkpoint for a fold atomically. Returns the path to best.pt."""
        if not is_global_zero():
            return ""

        d = self._fold_dir(fold_id)
        path = os.path.join(d, "best.pt")
        json_path = os.path.join(d, "best.json")

        state: Dict[str, Any] = {
            "fold_id": int(fold_id),
            "fold_name": str(fold_name),
            "best_val_metric": float(best_val_metric),
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
            "sched": scheduler_state,
            "cfg": self.cfg,
        }
        if extra:
            state.update(extra)

        tmp_fd, tmp_path = tempfile.mkstemp(dir=d, prefix="tmp_fold_", suffix=".pt")
        os.close(tmp_fd)
        torch.save(state, tmp_path)
        os.replace(tmp_path, path)

        # JSON sidecar carries metadata only. no model weights.
        try:
            meta = {
                "fold_id": int(fold_id),
                "fold_name": str(fold_name),
                "best_val_metric": float(best_val_metric),
                "extra": extra or {},
            }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, default=str)
        except Exception:
            pass

        return path
    
    def fold_dir(self, fold_id: int, fold_name: str) -> Path:
        out_dir = Path(str((self.cfg.get("trainer") or {}).get("output_dir", "runs"))).expanduser()
        d = out_dir / f"fold_{fold_id:02d}" / fold_name
        d.mkdir(parents=True, exist_ok=True)
        return d
