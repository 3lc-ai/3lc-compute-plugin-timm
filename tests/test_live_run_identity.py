# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Trainer run identity is published before progress, even if execution later fails."""

from __future__ import annotations

import sys
import threading
from types import ModuleType

import pytest
from tlc_plugin_sdk import JobContext

from tlc_plugin_timm import TimmPlugin


@pytest.mark.parametrize("mode", ["train", "collect"])
def test_trainer_run_is_published_while_the_job_is_still_running(monkeypatch, tmp_path, mode):
    events = []
    run_url = "s3://bucket/projects/Fire/runs/actual-unique-run"

    def execute(tables, params, callbacks):
        callbacks["on_run"](run_url)
        assert any(e.get("event") == "result" and e["run_url"] == run_url for e in events)
        callbacks["on_epoch"](1, 50, {})
        msg = "later execution failure"
        raise RuntimeError(msg)

    trainer = ModuleType("tlc_plugin_timm.trainer")
    trainer.train = execute
    trainer.collect = execute
    monkeypatch.setitem(sys.modules, "tlc_plugin_timm.trainer", trainer)
    ctx = JobContext(
        "live-run",
        {
            "project_config": {
                "name": "preset",
                "project_name": "Fire",
                "run_name": "requested-name",
                "model_name": "resnet18",
                "mode": mode,
                "image_column": "image",
                "label_column": "label",
                "use_latest": False,
                "train_table_url": "s3://bucket/table",
            }
        },
        tmp_path,
        sink=events.append,
        cancel_event=threading.Event(),
    )
    with pytest.raises(RuntimeError, match="later execution failure"):
        TimmPlugin().run_job(ctx)
    result = next(i for i, e in enumerate(events) if e.get("event") == "result")
    progress = next(i for i, e in enumerate(events) if e.get("event") == "progress")
    assert result < progress
