#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# =======================================================================
#  Project       : CiproSeg - LOSO U-Net Segmentation Framework
#  File          : src/ciproseg/core/hooks.py
#  Purpose       : Hook
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

from typing import Any, Dict, List, Optional


class Hook:
    """Base hook: all methods are no-ops. Subclass and override as needed."""

    def on_train_start(self, state: Dict[str, Any]) -> None: ...
    def on_train_end(self, state: Dict[str, Any]) -> None: ...
    def on_fold_start(self, state: Dict[str, Any]) -> None: ...
    def on_fold_end(self, state: Dict[str, Any]) -> None: ...
    def on_epoch_start(self, state: Dict[str, Any]) -> None: ...
    def on_epoch_end(self, state: Dict[str, Any]) -> None: ...
    def on_batch_start(self, state: Dict[str, Any]) -> None: ...
    def on_batch_end(self, state: Dict[str, Any]) -> None: ...
    def on_eval_start(self, state: Dict[str, Any]) -> None: ...
    def on_eval_end(self, state: Dict[str, Any]) -> None: ...


class HookList(Hook):
    """Aggregates multiple hooks and fans out each lifecycle call in order."""

    def __init__(self, hooks: Optional[List[Hook]] = None) -> None:
        self.hooks: List[Hook] = hooks or []

    def _dispatch(self, name: str, state: Dict[str, Any]) -> None:
        for h in self.hooks:
            getattr(h, name)(state)

    def on_train_start(self, state: Dict[str, Any]) -> None: self._dispatch("on_train_start", state)
    def on_train_end(self, state: Dict[str, Any]) -> None: self._dispatch("on_train_end", state)
    def on_fold_start(self, state: Dict[str, Any]) -> None: self._dispatch("on_fold_start", state)
    def on_fold_end(self, state: Dict[str, Any]) -> None: self._dispatch("on_fold_end", state)
    def on_epoch_start(self, state: Dict[str, Any]) -> None: self._dispatch("on_epoch_start", state)
    def on_epoch_end(self, state: Dict[str, Any]) -> None: self._dispatch("on_epoch_end", state)
    def on_batch_start(self, state: Dict[str, Any]) -> None: self._dispatch("on_batch_start", state)
    def on_batch_end(self, state: Dict[str, Any]) -> None: self._dispatch("on_batch_end", state)
    def on_eval_start(self, state: Dict[str, Any]) -> None: self._dispatch("on_eval_start", state)
    def on_eval_end(self, state: Dict[str, Any]) -> None: self._dispatch("on_eval_end", state)
    