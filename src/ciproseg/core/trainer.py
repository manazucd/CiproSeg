#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# =======================================================================
#  Project       : CiproSeg - LOSO U-Net Segmentation Framework
#  File          : src/ciproseg/core/trainer.py
#  Purpose       : Trainer
#
#  Copyright (c) 2026 Manaz Kaleel
#  Authors       : Manaz Kaleel <manaz.ucd@gmail.com>
#
#  Created on    : 2025-11-01
#  Last modified : 2026-02-26, MK
#  Version       : 0.1.0
#  License       : Apache-2.0
#  Repository    : https://github.com/manazucd/CiproSeg
# =======================================================================
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
import numpy as np

import torch, gc
import torch.nn as nn
from torch.utils.data import DataLoader

from ..models import build_model_from_cfg
from ..optim.optimizers import build_optimizer
from ..optim.schedulers import build_scheduler, SchedulerBundle
from ..utils.seed import seed_everything
from ..utils.checkpoint import Checkpointer
from ..data.collate import segmentation_collate_fn
from ..data.datasets.ciproseg_ecoli import (
    build_index,
    make_loso_folds,
    build_fold_splits,
    RandomCropSegDataset,
    default_crops_per_frame,
)
from .evaluator import evaluate_series_full
from .hooks import HookList
from ..visualization.plot_curves import Curve, FoldCurves, plot_fold_curves

_METRIC_KEYS = ("dice", "iou", "precision", "recall")


class Trainer:
    """
    LOSO single-GPU trainer for CiproSeg.

    Expected interface (consumed by Engine):
        Trainer(cfg, datamodule=None, hooks=None)
        .train(start_state=None)
        .best_metric : scalar set after all folds complete
        .last_metrics: mean test metrics across folds
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        datamodule: Any = None,
        hooks: Optional[HookList] = None,
    ) -> None:
        self.cfg = cfg
        self.datamodule = datamodule
        self.hooks = hooks or HookList([])
        self.ckpt = Checkpointer(cfg)

        dcfg = cfg["dataset"]
        self.dataset_root = dcfg["root"]
        self.train_frac = float(dcfg.get("train_frac", 0.8))
        self.patch = int(dcfg.get("patch", 256))
        self.stride = int(dcfg.get("stride", 128))

        tcfg = cfg.get("trainer") or {}
        self.epochs = int(tcfg.get("epochs", 40))
        self.batch_size = int(tcfg.get("batch_size", 8))
        self.num_workers = int(tcfg.get("num_workers", 2))
        self.seed = int(tcfg.get("seed", 1234))
        self.thr = float(tcfg.get("thr", 0.5))
        self.target_patches_per_epoch = int(tcfg.get("target_patches_per_epoch", 2000))
        self.deterministic = bool(tcfg.get("deterministic", False))
        self.benchmark = bool(tcfg.get("benchmark", True))

        force_cpu = bool(tcfg.get("cpu", False))
        self.device = "cuda" if (torch.cuda.is_available() and not force_cpu) else "cpu"

        mname = (self.cfg.get("model") or {}).get("name", "")
        req = 16 if mname == "unet4" else 4 
        if self.patch % req != 0:
            raise ValueError(f"patch must be divisible by {req} for model '{mname}'.")

        self.best_metric: float = float("-inf")
        self.last_metrics: Dict[str, float] = {}
        self.fold_results: List[Dict[str, Any]] = []

    def train(self, start_state: Optional[Dict[str, Any]] = None) -> None:
        _ = start_state
        run_dir = Path((self.cfg.get("trainer") or {}).get("output_dir", "runs"))
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "config.yaml", "w") as f:
            yaml.safe_dump(self.cfg, f, sort_keys=False)

        state: Dict[str, Any] = {"cfg": self.cfg}
        self.hooks.on_train_start(state)

        index = build_index(self.dataset_root)
        folds = make_loso_folds(index)

        all_fold_test: List[Dict[str, float]] = []
        global_best_val = float("-inf")
        self.fold_results = []

        for fold_id, (test_s, train_s) in enumerate(folds, start=1):
            torch.cuda.empty_cache()
            gc.collect()

            fold_name = f"fold{fold_id:02d}_test_{test_s}"
            fold_seed = self.seed + 1000 * fold_id

            train_frames, val_frames, test_frames = build_fold_splits(
                index=index,
                test_series=test_s,
                train_series=train_s,
                train_frac=self.train_frac,
            )

            seed_everything(fold_seed, deterministic=self.deterministic, benchmark=self.benchmark)

            model = build_model_from_cfg(self.cfg).to(self.device)
            criterion = nn.BCEWithLogitsLoss()

            K = default_crops_per_frame(
                num_train_frames=len(train_frames),
                target_patches_per_epoch=self.target_patches_per_epoch,
                kmin=4,
                kmax=32,
            )
            train_ds = RandomCropSegDataset(
                frames=train_frames,
                patch=self.patch,
                crops_per_frame=K,
                fg_frac=0.01,
                mode_probs=(0.6, 0.3, 0.1),
                do_augment=True,
                return_meta=True,
            )
            train_loader = DataLoader(
                train_ds,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=(self.device == "cuda"),
                collate_fn=segmentation_collate_fn,
            )

            opt = build_optimizer(model, self.cfg)
            sched_bundle: Optional[SchedulerBundle] = build_scheduler(
                opt,
                self.cfg,
                steps_per_epoch=len(train_loader),
                epochs=self.epochs,
            )

            state: Dict[str, Any] = {
                "cfg": self.cfg,
                "fold_id": fold_id,
                "fold_name": fold_name,
                "test_series": test_s,
                "n_train": len(train_frames),
                "n_val": len(val_frames),
                "n_test": len(test_frames),
                "patch": self.patch,
                "stride": self.stride,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "device": self.device,
                "optim": cfg_section(self.cfg, "optim"),
                "sched": cfg_section(self.cfg, "sched"),
                "lr": float(opt.param_groups[0].get("lr", 0.0)),
            }
            self.hooks.on_fold_start(state)

            best_val_dice = float("-inf")
            best_state: Optional[Dict[str, torch.Tensor]] = None

            loss_epochs: List[int] = []
            loss_train: List[float] = []
            loss_val: List[Optional[float]] = []
            dice_epochs: List[int] = []
            dice_val: List[float] = []

            fold_dir = self.ckpt.fold_dir(fold_id, fold_name)
            loss_png = fold_dir / "loss.png"
            dice_png = fold_dir / "dice.png"

            for ep in range(1, self.epochs + 1):
                state["epoch"] = ep
                self.hooks.on_epoch_start(state)

                model.train()
                running_loss = 0.0

                for batch in train_loader:
                    self.hooks.on_batch_start(state)

                    xb = batch["x"].to(self.device, non_blocking=True)
                    yb = batch["y"].to(self.device, non_blocking=True)

                    opt.zero_grad(set_to_none=True)
                    loss = criterion(model(xb), yb)
                    loss.backward()
                    opt.step()

                    if sched_bundle is not None and sched_bundle.interval == "step":
                        sched_bundle.scheduler.step()

                    running_loss += float(loss.detach().cpu())
                    self.hooks.on_batch_end(state)

                if sched_bundle is not None and sched_bundle.interval == "epoch":
                    sched_bundle.scheduler.step()

                state["train_loss"] = running_loss / max(1, len(train_loader))
                state["lr"] = float(opt.param_groups[0].get("lr", 0.0))
                state["is_eval_epoch"] = (ep % 10 == 0) or (ep == self.epochs)

                if (ep % 10 == 0) or (ep == self.epochs):

                    with torch.inference_mode():
                        val_metrics = evaluate_series_full(
                            model,
                            val_frames,
                            patch=self.patch,
                            stride=self.patch,
                            device=self.device,
                            thr=self.thr,
                            compute_loss=True,
                        )
                    torch.cuda.empty_cache()
                    gc.collect()

                    val_dice = float(val_metrics["dice"])
                    state["val_metrics"] = val_metrics

                    loss_epochs.append(ep)
                    loss_train.append(float(state["train_loss"]))
                    loss_val.append(_extract_val_loss(val_metrics))
                    dice_epochs.append(ep)
                    dice_val.append(val_dice)

                    if val_dice > best_val_dice:
                        best_val_dice = val_dice
                        best_state = {
                            k: v.detach().cpu().clone()
                            for k, v in model.state_dict().items()
                        }

                        sched_state_best = None
                        if sched_bundle is not None:
                            try:
                                sched_state_best = sched_bundle.scheduler.state_dict()
                            except Exception:
                                sched_state_best = None

                        self.ckpt.save_fold_best(
                            fold_id=fold_id,
                            fold_name=fold_name,
                            model=model,
                            optimizer=opt,
                            scheduler_state=sched_state_best,
                            best_val_metric=best_val_dice,
                            extra={
                                "fold_seed": fold_seed,
                                "epoch": ep,
                                "val_best_metrics": {"dice": best_val_dice},
                            },
                        )

                    sched_state_live = None
                    if sched_bundle is not None:
                        try:
                            sched_state_live = sched_bundle.scheduler.state_dict()
                        except Exception:
                            sched_state_live = None

                    self.ckpt.save_fold_last(
                        fold_id=fold_id,
                        fold_name=fold_name,
                        epoch=ep,
                        model=model,
                        optimizer=opt,
                        scheduler_state=sched_state_live,
                        best_val_metric=best_val_dice,
                        extra={
                            "fold_seed": fold_seed,
                            "lr": float(opt.param_groups[0].get("lr", 0.0)),
                            "train_loss": float(state["train_loss"]),
                            "val_metrics": val_metrics,
                        },
                    )

                    curves_loss: Dict[str, Curve] = {
                        "train_loss": Curve(
                            name="train_loss",
                            x=[float(e) for e in loss_epochs],
                            y=[float(v) for v in loss_train],
                        ),
                    }
                    xs_v, ys_v = [], []
                    for e, v in zip(loss_epochs, loss_val):
                        if v is None:
                            continue
                        xs_v.append(float(e))
                        ys_v.append(float(v))
                    if ys_v:
                        curves_loss["val_loss"] = Curve(name="val_loss", x=xs_v, y=ys_v)

                    fc_loss = FoldCurves(fold_name=fold_name, curves=curves_loss)
                    plot_fold_curves(
                        fc_loss,
                        metrics=("train_loss", "val_loss"),
                        title=f"{fold_name} | loss",
                        show=False,
                        savepath=str(loss_png),
                    )

                    fc_dice = FoldCurves(
                        fold_name=fold_name,
                        curves={
                            "val_dice": Curve(
                                name="val_dice",
                                x=[float(e) for e in dice_epochs],
                                y=[float(v) for v in dice_val],
                            ),
                        },
                    )
                    plot_fold_curves(
                        fc_dice,
                        metrics=("val_dice",),
                        title=f"{fold_name} | dice",
                        show=False,
                        savepath=str(dice_png),
                    )

                state["best_val_metric"] = best_val_dice
                self.hooks.on_epoch_end(state)

            if best_state is not None:
                model.load_state_dict(best_state)

            with torch.inference_mode():
                test_metrics = evaluate_series_full(
                    model,
                    test_frames,
                    patch=self.patch,
                    stride=self.stride,
                    device=self.device,
                    thr=self.thr,
                )

            state["test_metrics"] = test_metrics
            self.hooks.on_fold_end(state)

            self.fold_results.append({
                "fold_id": fold_id,
                "fold_name": fold_name,
                "test_series": test_s,
                "best_val_dice": float(best_val_dice),
                "test_metrics": test_metrics,
            })

            all_fold_test.append(test_metrics)
            global_best_val = max(global_best_val, best_val_dice)

        self.best_metric = float(global_best_val)
        self.last_metrics = (
            {k: float(np.mean([m[k] for m in all_fold_test])) for k in _METRIC_KEYS}
            if all_fold_test
            else dict.fromkeys(_METRIC_KEYS, 0.0)
        )
        self.ckpt.save_run_summary(self, metric_value=self.best_metric)


def _extract_val_loss(val_metrics: Dict[str, Any]) -> Optional[float]:
    """
    Try common keys for validation loss returned by evaluate_series_full().
    Kept tolerant to avoid crashing shall loss not provided.
    """
    for k in ("loss", "val_loss", "bce", "bce_loss"):
        if k in val_metrics:
            try:
                return float(val_metrics[k])
            except Exception:
                pass
    return None


def cfg_section(cfg: Dict[str, Any], key: str) -> Dict[str, Any]:
    """Return a config sub-section safely, defaulting to an empty dict."""
    return cfg.get(key) or {}
