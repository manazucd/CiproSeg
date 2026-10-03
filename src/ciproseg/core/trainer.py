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
#  Last modified : 2026-10-03, MK
#  Version       : 0.1.0
#  License       : Apache-2.0
#  Repository    : https://github.com/manazucd/CiproSeg
# =======================================================================
from __future__ import annotations

import gc
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

import torch
from torch.utils.data import DataLoader

from ..models import build_model_from_cfg
from ..optim.optimizers import build_optimizer
from ..optim.losses import build_criterion
from ..optim.schedulers import build_scheduler, SchedulerBundle
from ..utils.seed import seed_everything
from ..utils.checkpoint import Checkpointer
from ..utils.logger import get_logger
from ..data.collate import segmentation_collate_fn
from ..data.datasets.ciproseg_ecoli import (
    build_index,
    make_loso_folds,
    build_fold_splits,
    RandomCropSegDataset,
    default_crops_per_frame,
)
from .evaluator import (
    evaluate_series_full,
    evaluate_with_threshold_sweep,
    evaluate_with_object_metrics,
    find_best_threshold,
)
from .hooks import HookList
from ..visualization.plot_curves import Curve, FoldCurves, plot_fold_curves


_METRIC_KEYS = ("dice", "iou", "precision", "recall")

# Validation cadence: the trainer evaluates the model every N epochs
# rather than every epoch. Keeps wallclock cost down on long runs.
_VAL_EVERY_N_EPOCHS = 10

# Default threshold grid swept on val before running the final test.
# Range chosen to bracket the empirically observed CiproSeg optima.


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
        self.logger = get_logger()

        self.run_dir = resolve_run_output_dir(self.cfg)
        self.ckpt = Checkpointer(self.cfg)

        dcfg = cfg["dataset"]
        self.dataset_root = dcfg["root"]
        self.train_frac = float(dcfg.get("train_frac", 0.8))
        self.patch = int(dcfg.get("patch", 256))
        self.stride = int(dcfg.get("stride", 128))

        tcfg = cfg.get("trainer") or {}
        vcfg = cfg.get("validation") or {}

        self.epochs = int(tcfg.get("epochs", 40))
        self.batch_size = int(tcfg.get("batch_size", 8))
        self.num_workers = int(tcfg.get("num_workers", 2))
        self.seed = int(tcfg.get("seed", 1234))

        self.checkpoint_threshold = float(
            vcfg.get("checkpoint_threshold", 0.5)
        )

        if not (0.0 < self.checkpoint_threshold < 1.0):
            raise ValueError(
                "validation.checkpoint_threshold must be in (0, 1), "
                f"got {self.checkpoint_threshold}"
            )

        self.target_patches_per_epoch = int(tcfg.get("target_patches_per_epoch", 2000))
        self.deterministic = bool(tcfg.get("deterministic", False))
        self.benchmark = bool(tcfg.get("benchmark", True))

        # New options. All default to current behaviour, so existing
        # configs continue to run unchanged.
        self.early_stop_patience = tcfg.get("early_stop_patience", None)
        if self.early_stop_patience is not None:
            self.early_stop_patience = int(self.early_stop_patience)
            if self.early_stop_patience <= 0:
                raise ValueError(
                    f"early_stop_patience must be > 0, got {self.early_stop_patience}"
                )

        self.use_amp = bool(tcfg.get("amp", False))

        self.threshold_sweep_enabled = bool(
        vcfg.get("threshold_sweep", False)
        )

        self.sweeping_thresholds = self._resolve_sweeping_thresholds(
            enabled=self.threshold_sweep_enabled,
            thresholds_cfg=vcfg.get("sweeping_thresholds", None),
        )

        self.test_tta = bool(tcfg.get("test_tta", False))
        self.compute_test_object_metrics = bool(tcfg.get("compute_object_metrics", False))

        force_cpu = bool(tcfg.get("cpu", False))
        use_cuda = (not force_cpu) and torch.cuda.is_available() and (torch.cuda.device_count() > 0)
        self.device = "cuda" if use_cuda else "cpu"

        if self.use_amp and self.device != "cuda":
            self.logger.warning(
                "[Trainer] AMP requested but device is CPU; disabling AMP."
            )
            self.use_amp = False

        mname = (self.cfg.get("model") or {}).get("name", "")
        req = 16 if mname == "unet4" else 4
        if self.patch % req != 0:
            raise ValueError(f"patch must be divisible by {req} for model '{mname}'.")

        self.best_metric: float = float("-inf")
        self.last_metrics: Dict[str, float] = {}
        self.fold_results: List[Dict[str, Any]] = []

    @staticmethod
    def _resolve_sweeping_thresholds(
        enabled: bool,
        thresholds_cfg: Any,
    ) -> Optional[Sequence[float]]:
        """
        Validate the thresholds used for the post training
        validation threshold sweep.

        If threshold_sweep is False, no sweep is performed and the
        checkpoint threshold is used.

        If threshold_sweep is True, sweeping_thresholds must contain
        one or more values in the open interval (0, 1).

        The number of thresholds is unrestricted.
        """
        if not enabled:
            return None

        if thresholds_cfg is None:
            raise ValueError(
                "validation.threshold_sweep is true, but "
                "validation.sweeping_thresholds is not defined"
            )

        if not isinstance(thresholds_cfg, (list, tuple)):
            raise ValueError(
                "validation.sweeping_thresholds must be a list of numbers"
            )

        if len(thresholds_cfg) == 0:
            raise ValueError(
                "validation.sweeping_thresholds cannot be empty"
            )

        values = tuple(float(v) for v in thresholds_cfg)

        for v in values:
            if not (0.0 < v < 1.0):
                raise ValueError(
                    "validation.sweeping_thresholds values must be "
                    f"in (0, 1), got {v}"
                )

        return values

    def train(self, start_state: Optional[Dict[str, Any]] = None) -> None:
        _ = start_state

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

            fold_name = f"fold_{fold_id:02d}_test_{test_s}"
            fold_seed = self.seed + 1000 * fold_id

            train_frames, val_frames, test_frames = build_fold_splits(
                index=index,
                test_series=test_s,
                train_series=train_s,
                train_frac=self.train_frac,
            )

            seed_everything(fold_seed, deterministic=self.deterministic, benchmark=self.benchmark)

            model = build_model_from_cfg(self.cfg).to(self.device)
            criterion = build_criterion(self.cfg)

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

            # AMP scaler. Disabled scaler is a cheap no-op so we always
            # construct one and let `enabled` toggle the behaviour.
            scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

            state = {
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
                "amp": self.use_amp,
                "early_stop_patience": self.early_stop_patience,
                "optim": cfg_section(self.cfg, "optim"),
                "sched": cfg_section(self.cfg, "sched"),
                "lr": float(opt.param_groups[0].get("lr", 0.0)),
            }
            self.hooks.on_fold_start(state)

            best_val_dice = float("-inf")
            best_state: Optional[Dict[str, torch.Tensor]] = None
            best_epoch = 0
            stopped_early = False

            loss_epochs: List[int] = []
            loss_train: List[float] = []
            loss_val: List[Optional[float]] = []
            dice_epochs: List[int] = []
            dice_val: List[float] = []

            fold_dir = self.ckpt.fold_dir(fold_name)
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

                    with torch.amp.autocast(device_type="cuda", enabled=self.use_amp):
                        logits = model(xb)
                        loss = criterion(logits, yb)

                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()

                    if sched_bundle is not None and sched_bundle.interval == "step":
                        sched_bundle.scheduler.step()

                    running_loss += float(loss.detach().cpu())
                    self.hooks.on_batch_end(state)

                if sched_bundle is not None and sched_bundle.interval == "epoch":
                    sched_bundle.scheduler.step()

                state["train_loss"] = running_loss / max(1, len(train_loader))
                state["lr"] = float(opt.param_groups[0].get("lr", 0.0))

                is_eval_epoch = (ep % _VAL_EVERY_N_EPOCHS == 0) or (ep == self.epochs)
                state["is_eval_epoch"] = is_eval_epoch

                if is_eval_epoch:

                    with torch.inference_mode():
                        val_metrics = evaluate_series_full(
                            model,
                            val_frames,
                            patch=self.patch,
                            stride=self.stride,
                            device=self.device,
                            thr=self.checkpoint_threshold,
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

                    improved = val_dice > best_val_dice
                    if improved:
                        best_val_dice = val_dice
                        best_epoch = ep
                        best_state = {
                            k: v.detach().cpu().clone()
                            for k, v in model.state_dict().items()
                        }

                        sched_state_best = self._safe_scheduler_state(sched_bundle)

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

                    self.ckpt.save_fold_last(
                        fold_id=fold_id,
                        fold_name=fold_name,
                        epoch=ep,
                        model=model,
                        optimizer=opt,
                        scheduler_state=self._safe_scheduler_state(sched_bundle),
                        best_val_metric=best_val_dice,
                        extra={
                            "fold_seed": fold_seed,
                            "lr": float(opt.param_groups[0].get("lr", 0.0)),
                            "train_loss": float(state["train_loss"]),
                            "val_metrics": val_metrics,
                        },
                    )

                    self._update_curves(
                        fold_name=fold_name,
                        loss_png=loss_png,
                        dice_png=dice_png,
                        loss_epochs=loss_epochs,
                        loss_train=loss_train,
                        loss_val=loss_val,
                        dice_epochs=dice_epochs,
                        dice_val=dice_val,
                    )

                    if (
                        self.early_stop_patience is not None
                        and (ep - best_epoch) >= self.early_stop_patience
                    ):
                        self.logger.info(
                            "[Trainer] %s: early stopping at epoch %d (best=%d, patience=%d)",
                            fold_name,
                            ep,
                            best_epoch,
                            self.early_stop_patience,
                        )
                        stopped_early = True
                        state["best_val_metric"] = best_val_dice
                        self.hooks.on_epoch_end(state)
                        break

                state["best_val_metric"] = best_val_dice
                self.hooks.on_epoch_end(state)

            if best_state is not None:
                model.load_state_dict(best_state)

            chosen_thr, val_thr_summary = self._select_threshold(
                model, val_frames
            )

            if self.compute_test_object_metrics:
                with torch.inference_mode():
                    test_metrics = evaluate_with_object_metrics(
                        model,
                        test_frames,
                        patch=self.patch,
                        stride=self.stride,
                        device=self.device,
                        thr=chosen_thr,
                        use_tta=self.test_tta,
                    )
            else:
                with torch.inference_mode():
                    test_metrics = evaluate_series_full(
                        model,
                        test_frames,
                        patch=self.patch,
                        stride=self.stride,
                        device=self.device,
                        thr=chosen_thr,
                        use_tta=self.test_tta,
                    )

            test_metrics["stopped_early"] = bool(stopped_early)
            test_metrics["best_checkpoint_epoch"] = int(best_epoch)

            state["test_metrics"] = test_metrics
            self.hooks.on_fold_end(state)

            selected_thr_key = f"{chosen_thr:.2f}"

            selected_metric_value = None
            if val_thr_summary and selected_thr_key in val_thr_summary:
                selected_metric_value = float(
                    val_thr_summary[selected_thr_key]["dice"]
                )

            fold_record = {
                "fold_id": fold_id,
                "fold_name": fold_name,
                "test_series": test_s,

                "best_checkpoint_validation_dice": float(best_val_dice),
                "best_checkpoint_epoch": int(best_epoch),
                "stopped_early": bool(stopped_early),

                "threshold_selection": {
                    "selection_set": "validation",
                    "selection_metric": "dice",
                    "selected_threshold": float(chosen_thr),
                    "selected_metric_value": selected_metric_value,
                    "candidate_thresholds": list(
                        self.sweeping_thresholds or []
                    ),
                    "metrics_by_threshold": val_thr_summary,
                },

                "test_metrics": test_metrics,
            }         
            self.fold_results.append(fold_record)

            all_fold_test.append(test_metrics)
            global_best_val = max(global_best_val, best_val_dice)

            self.logger.info(
                "[Trainer] %s done: best_val=%.4f @ ep %d, thr=%.2f, "
                "test_dice=%.4f, test_iou=%.4f",
                fold_name,
                best_val_dice,
                best_epoch,
                chosen_thr,
                float(test_metrics.get("dice", 0.0)),
                float(test_metrics.get("iou", 0.0)),
            )

        self.best_metric = float(global_best_val)
        self.last_metrics = (
            {k: float(np.mean([m[k] for m in all_fold_test])) for k in _METRIC_KEYS}
            if all_fold_test
            else dict.fromkeys(_METRIC_KEYS, 0.0)
        )
        self.ckpt.save_run_summary(self, metric_value=self.best_metric)

    def _select_threshold(
        self,
        model: torch.nn.Module,
        val_frames: List[Any],
    ):
        """
        Select the final binary mask threshold using the validation set.

        If validation.threshold_sweep is enabled, evaluate all thresholds
        listed in validation.sweeping_thresholds and select the threshold
        with the highest mean validation Dice.

        Otherwise, use validation.checkpoint_threshold.
        """
        if self.sweeping_thresholds is None:
            return float(self.checkpoint_threshold), {}

        with torch.inference_mode():
            sweep = evaluate_with_threshold_sweep(
                model,
                val_frames,
                patch=self.patch,
                stride=self.stride,
                device=self.device,
                thresholds=self.sweeping_thresholds,
                use_tta=self.test_tta,
            )
        best_thr, _ = find_best_threshold(sweep, metric="dice")
        summary = {f"{t:.2f}": sweep[t] for t in sorted(sweep.keys())}
        return float(best_thr), summary

    def _update_curves(
        self,
        *,
        fold_name: str,
        loss_png: Path,
        dice_png: Path,
        loss_epochs: List[int],
        loss_train: List[float],
        loss_val: List[Optional[float]],
        dice_epochs: List[int],
        dice_val: List[float],
    ) -> None:
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

    @staticmethod
    def _safe_scheduler_state(sched_bundle: Optional[SchedulerBundle]):
        if sched_bundle is None:
            return None
        try:
            return sched_bundle.scheduler.state_dict()
        except Exception:
            return None


def _extract_val_loss(val_metrics: Dict[str, Any]) -> Optional[float]:
    """
    Try common keys for validation loss returned by evaluate_series_full().
    Kept tolerant to avoid crashing should loss not be provided.
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


def _safe_name(s: str) -> str:
    """Filesystem-safe short name."""
    keep = []
    for ch in str(s).strip().replace(" ", "_"):
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
    return "".join(keep) or "model"


def resolve_run_output_dir(cfg: Dict[str, Any]) -> Path:
    """
    Resolve a unique, timestamped run directory and update cfg in place.

    Layout:  <base_output_dir> / <model_name> / <timestamp>
    Example: runs/unet4/2026-04-29_153000
    """
    tcfg = cfg.setdefault("trainer", {})

    if tcfg.get("_output_dir_resolved", False):
        return Path(str(tcfg["output_dir"]))

    mcfg = cfg.get("model") or {}
    base_dir = Path(str(tcfg.get("output_dir", "runs"))).expanduser()
    model_name = _safe_name(mcfg.get("name", "model"))
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    run_dir = base_dir / model_name / timestamp
    tcfg["output_dir"] = str(run_dir)
    tcfg["_output_dir_resolved"] = True

    return run_dir
