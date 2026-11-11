# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Persist exact training inputs and settings alongside collected metrics."""

from __future__ import annotations

from typing import Any


def record_run_metadata(run: Any, train_table: Any, val_table: Any, params: dict[str, Any]) -> None:
    """Keep lineage on the original tables, before augmentation or sampling views."""
    run.add_input_table(train_table)
    if str(val_table.url) != str(train_table.url):
        run.add_input_table(val_table)
    run.set_parameters(params)
