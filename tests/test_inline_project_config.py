# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""The inline project_config convention: a run body must be self-contained so a
remote GPU worker (which has no controller-local config store) can run it."""

from __future__ import annotations

from tlc_plugin_timm import _config_from_inline
from tlc_plugin_timm.config_store import TimmConfig

_LOGS: list[str] = []


def _log(msg: str) -> None:
    _LOGS.append(msg)


def test_valid_inline_config_builds_config() -> None:
    raw = {
        "id": "c1",
        "name": "cfg",
        "project_name": "proj",
        "model_name": "resnet50",
        "train_table_url": "s3://b/t",
        "mode": "train",
        "params": {"epochs": "2"},
    }
    config = _config_from_inline(raw, log=_log)
    assert isinstance(config, TimmConfig)
    assert config.id == "c1"
    assert config.train_table_url == "s3://b/t"
    assert config.params == {"epochs": "2"}


def test_unknown_keys_are_dropped_not_fatal() -> None:
    config = _config_from_inline({"id": "c2", "future_field": 1, "params": {}}, log=_log)
    assert isinstance(config, TimmConfig)
    assert config.id == "c2"


def test_non_dict_and_empty_resolve_to_none() -> None:
    assert _config_from_inline(None, log=_log) is None
    assert _config_from_inline("", log=_log) is None
    assert _config_from_inline({}, log=_log) is None
    assert _config_from_inline(["not", "a", "dict"], log=_log) is None
