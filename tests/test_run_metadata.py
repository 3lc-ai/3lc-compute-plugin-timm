# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""New runs retain exact revisions and settings in their on-disk record."""

import json

import pytest

from tlc_plugin_timm.run_metadata import record_run_metadata


@pytest.mark.parametrize("same_table", [False, True])
def test_inputs_and_settings_are_saved_with_native_run(tmp_path, same_table):
    import tlc

    tables = []
    for split in ("train", "validation"):
        writer = tlc.TableWriter(
            root_url=str(tmp_path),
            project_name="P",
            dataset_name=split,
            table_name="exact-revision",
            schema={"label": tlc.schemas.Int64Schema()},
        )
        writer.add_row({"label": 1})
        tables.append(writer.finalize())
    run = tlc.init(root_url=str(tmp_path), project_name="P", run_name="metadata")
    record_run_metadata(
        run,
        tables[0],
        tables[0] if same_table else tables[1],
        {"model_name": "efficientnet_b0", "epochs": 1, "image_size": 96},
    )
    record = json.loads((tmp_path / "P/runs/metadata/object.3lc.json").read_text())
    inputs = record["constants"]["inputs"]
    assert len(inputs) == (1 if same_table else 2)
    assert all(item["input_table_url"].endswith("/tables/exact-revision") for item in inputs)
    assert all(item["num_examples"] == 1 for item in inputs)
    assert record["constants"]["parameters"]["image_size"] == 96
