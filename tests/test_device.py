# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Explicit CPU execution never consults CUDA availability."""

import pytest


def test_explicit_device_is_respected(monkeypatch):
    torch = pytest.importorskip("torch")
    from tlc_plugin_timm.trainer import _device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("CPU must not probe CUDA"))
    assert str(_device("cpu")) == "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert str(_device("auto")) == "cuda"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert str(_device(None)) == "cpu"
    with pytest.raises(ValueError, match="not available"):
        _device("cuda")
    with pytest.raises(ValueError, match="must be"):
        _device("unrecognized")
