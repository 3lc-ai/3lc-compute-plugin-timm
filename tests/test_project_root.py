# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Runs are created under the project root the job carries (``ctx.project_root_url``)."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest
import torch.nn as nn
from tlc_plugin_sdk import JobContext

import tlc_plugin_timm.trainer as trainer
from tlc_plugin_timm import TimmPlugin

_INLINE_CONFIG = {
    "id": "c1",
    "project_name": "proj",
    "model_name": "resnet18",
    "train_table_url": "s3://b/t",
    "mode": "train",
    "params": {},
}


class _Stop(Exception):
    """Raised by the patched ``tlc.init`` so the trainer stops right after creating the run."""


def _run_job_params(monkeypatch: pytest.MonkeyPatch, ctx: Any) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def fake_train(tables: dict[str, Any], params: dict[str, Any], callbacks: dict[str, Any]) -> dict[str, Any]:
        captured.update(params)
        return {}

    monkeypatch.setattr(trainer, "train", fake_train)
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


class _OldSdkContext:
    """A context from an SDK that predates ``project_root_url``."""

    job_id = "j2"
    cancelled = False

    def __init__(self) -> None:
        self.params = {"project_config": _INLINE_CONFIG}

    def __getattr__(self, name: str) -> Any:
        if name == "project_root_url":
            raise AttributeError(name)
        return lambda *args, **kwargs: None


def test_run_job_without_the_property_leaves_the_root_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    params = _run_job_params(monkeypatch, _OldSdkContext())
    assert params["_project_root_url"] == ""


class _FakeTable:
    def get_value_map(self, column: str) -> dict[int, Any]:
        return {0: "a", 1: "b"}

    def with_transform(self, transform: Any) -> _FakeTable:
        return self


def _collect_init_kwargs(monkeypatch: pytest.MonkeyPatch, params: dict[str, Any]) -> dict[str, Any]:
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
