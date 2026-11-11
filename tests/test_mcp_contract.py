# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Agent training has the same frozen inputs and embedding options as the UI."""

from pathlib import Path

import pytest
import tomllib
from jsonschema import ValidationError, validate
from tlc_plugin_sdk.mcp_tools import load_mcp_section, parse_mcp_tools, render_job_body


def test_mcp_training_freezes_revisions_and_retains_embedding_settings():
    manifest = tomllib.loads((Path(__file__).parents[1] / "src/tlc_plugin_timm/plugin.toml").read_text())
    tools = parse_mcp_tools(
        load_mcp_section(manifest["mcp"], Path(__file__).parents[1] / "src/tlc_plugin_timm")["tools"]
    )
    training = next(tool for tool in tools if tool.name == "train")
    args = {
        "model": "efficientnet_b0",
        "device": "cpu",
        "num_workers": 0,
        "train_table_url": "s3://p/train/initial",
        "val_table_url": "s3://p/val/initial",
        "epochs": 2,
        "image_size": 96,
        "write_full_embeddings": True,
        "image_embeddings_dim": 2,
        "image_column": "img",
        "label_column": "label",
    }
    validate(args, training.input_schema)
    body = render_job_body(training.job["body"], args)
    from tlc_plugin_timm.job_contract import contract

    config = contract().normalize(body["project_config"])
    assert config["model_name"] == "efficientnet_b0"
    assert config["use_latest"] is False
    assert config["params"]["device"] == "cpu"
    assert config["params"]["num_workers"] == 0
    with pytest.raises(ValidationError):
        validate({**args, "num_workers": -1}, training.input_schema)
    assert config["train_table_url"] == args["train_table_url"]
    assert config["params"]["write_full_embeddings"] is True
    assert config["params"]["collection_disable"] is False
    assert config["params"]["image_embeddings_dim"] == 2
    assert config["image_column"] == "img"
    # Column names come from the native table inspection, never guessed by MCP.
    minimal = {
        "model": "efficientnet_b0",
        "train_table_url": "s3://p/train/initial",
        "image_column": "img",
        "label_column": "label",
    }
    validate(minimal, training.input_schema)
    for column in ("image_column", "label_column"):
        with pytest.raises(ValidationError):
            validate({key: value for key, value in minimal.items() if key != column}, training.input_schema)
    normalized = contract().normalize(render_job_body(training.job["body"], minimal)["project_config"])
    assert normalized["params"]["epochs"] == 20
    assert normalized["use_latest"] is False
