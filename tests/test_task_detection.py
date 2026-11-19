# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""A train table that cannot be read is an error the UI can gate Start on, not just an unknown task."""

from __future__ import annotations

from pathlib import Path

from tlc_plugin_timm.task_detection import detect_task


def test_an_unreadable_table_answers_with_an_error(tmp_path: Path) -> None:
    result = detect_task(str(tmp_path / "no-such-table"))
    assert result["task"] == "unknown"
    assert result["error"].startswith("Could not load table: ")
    assert result["details"] == result["error"], "older pages show the details"


def test_metadata_includes_source_project_identity(monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace

    tlc = ModuleType("tlc")
    table = object()
    tlc.Table = SimpleNamespace(from_url=lambda _: table)
    monkeypatch.setitem(sys.modules, "tlc", tlc)
    monkeypatch.setattr(
        "tlc_plugin_sdk.shared.modality.detect_modality_from_table",
        lambda _: SimpleNamespace(
            modality="classification",
            image_column="image",
            label_column="label",
            class_names={0: "cat"},
            num_classes=1,
        ),
    )
    context = {"project_name": "A", "project_url": "s3://bucket/A", "project_root_url": "s3://bucket"}
    monkeypatch.setattr("tlc_plugin_sdk.shared.table_context.table_project_context", lambda _: context)
    result = detect_task("a")
    assert result.items() >= context.items()
