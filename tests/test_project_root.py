# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Runs are created under the project root the job carries (``ctx.project_root_url``)."""

from __future__ import annotations

import sys
import threading
import types
from pathlib import Path
from typing import Any

import pytest
from tlc_plugin_sdk import JobContext

from tlc_plugin_timm import TimmPlugin

_INLINE_CONFIG = {
    "id": "c1",
    "project_name": "proj",
    "model_name": "resnet18",
    "train_table_url": "s3://b/t",
    "mode": "train",
    "image_column": "image",
    "label_column": "label",
    "params": {},
}


class _Stop(Exception):
    """Raised by the patched ``tlc.init`` so the trainer stops right after creating the run."""


def _run_job_params(monkeypatch: pytest.MonkeyPatch, ctx: Any) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def fake_train(tables: dict[str, Any], params: dict[str, Any], callbacks: dict[str, Any]) -> dict[str, Any]:
        captured.update(params)
        return {}

    # The trainer imports torch at module level; CI installs the plugin without it. run_job imports the
    # trainer lazily, so a stand-in module is enough to see the params it hands over.
    stub = types.ModuleType("tlc_plugin_timm.trainer")
    vars(stub)["train"] = fake_train
    vars(stub)["collect"] = fake_train
    monkeypatch.setitem(sys.modules, "tlc_plugin_timm.trainer", stub)
    TimmPlugin().run_job(ctx)
    return captured


def test_run_job_hands_the_stamped_root_to_the_trainer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ctx = JobContext(
        "j1",
        {"project_config": _INLINE_CONFIG, "project_root_url": "s3://bucket/root/"},
        tmp_path,
        sink=lambda _event: None,
        cancel_event=threading.Event(),
    )
    params = _run_job_params(monkeypatch, ctx)
    assert params["_project_root_url"] == "s3://bucket/root"


class _NoRootContext:
    """A context whose worker cannot name a root (no stamped key, no ``tlc`` root)."""

    job_id = "j2"
    cancelled = False
    project_root_url = ""

    def __init__(self) -> None:
        self.params = {"project_config": _INLINE_CONFIG}

    def __getattr__(self, name: str) -> Any:
        return lambda *args, **kwargs: None


def test_run_job_without_a_root_leaves_it_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    params = _run_job_params(monkeypatch, _NoRootContext())
    assert params["_project_root_url"] == ""


def test_training_progress_moves_with_batches_and_does_not_regress_at_validation(monkeypatch, tmp_path):
    events = []
    ticks = iter(range(0, 10000, 10))
    monkeypatch.setattr("time.monotonic", lambda: next(ticks))

    def fake_train(tables, params, callbacks):
        for phase, fraction in (("train", 0.5), ("train", 1.0), ("val", 0.5), ("val", 1.0)):
            callbacks["on_epoch"](1, 2, {"phase": phase, "batch_frac": fraction, "batch": 1, "total_batches": 2})
        callbacks["on_epoch"](1, 2, {"val_acc": 50})
        callbacks["on_epoch"](2, 2, {"phase": "train", "batch_frac": 0.5, "batch": 1, "total_batches": 2})
        callbacks["on_status"]("Fitting umap on train embeddings: (1000, 1280) → 2D")
        return {}

    stub = types.ModuleType("tlc_plugin_timm.trainer")
    vars(stub)["train"] = fake_train
    vars(stub)["collect"] = fake_train
    monkeypatch.setitem(sys.modules, "tlc_plugin_timm.trainer", stub)
    ctx = JobContext(
        "progress",
        {"project_config": _INLINE_CONFIG, "project_root_url": str(tmp_path)},
        tmp_path,
        sink=events.append,
        cancel_event=threading.Event(),
    )
    TimmPlugin().run_job(ctx)
    progress = [
        event for event in events if event["event"] == "progress" and str(event.get("label", "")).startswith("Epoch")
    ]
    assert [event["percent"] for event in progress] == [20, 40, 45, 50, 50, 70]
    assert "batch 1/2" in progress[0]["label"]
    assert progress[-1]["timing"]["eta_s"] > 0
    collection = [event for event in events if event["event"] == "progress" and "Fitting" in event.get("label", "")]
    assert collection[0]["percent"] == -1
    assert "umap" in collection[0]["label"]


class _FakeTable:
    def get_value_map(self, column: str) -> dict[int, Any]:
        return {0: "a", 1: "b"}

    def with_transform(self, transform: Any) -> _FakeTable:
        return self


def _collect_init_kwargs(monkeypatch: pytest.MonkeyPatch, params: dict[str, Any]) -> dict[str, Any]:
    nn = pytest.importorskip("torch.nn")
    trainer = pytest.importorskip("tlc_plugin_timm.trainer")
    import timm
    import timm.data
    import tlc

    seen: dict[str, Any] = {}

    def fake_init(**kwargs: Any) -> Any:
        seen.update(kwargs)
        raise _Stop

    model = nn.Linear(1, 1)
    model.pretrained_cfg = {}
    monkeypatch.setattr(tlc.Table, "from_url", staticmethod(lambda url: _FakeTable()))
    monkeypatch.setattr(tlc, "init", fake_init)
    monkeypatch.setattr(timm, "create_model", lambda *args, **kwargs: model)
    monkeypatch.setattr(timm.data, "resolve_data_config", lambda cfg: {})
    monkeypatch.setattr(timm.data, "create_transform", lambda **kwargs: None)
    with pytest.raises(_Stop):
        trainer.collect({"train": "s3://b/t", "val": None}, {"_project_name": "proj", **params}, {})
    return seen


def test_collect_creates_the_run_under_the_job_root(monkeypatch: pytest.MonkeyPatch) -> None:
    kwargs = _collect_init_kwargs(monkeypatch, {"_project_root_url": "s3://bucket/root"})
    assert kwargs["root_url"] == "s3://bucket/root"
    assert kwargs["project_name"] == "proj"


def test_collect_without_a_root_keeps_the_tlc_default(monkeypatch: pytest.MonkeyPatch) -> None:
    kwargs = _collect_init_kwargs(monkeypatch, {})
    assert kwargs["root_url"] is None
