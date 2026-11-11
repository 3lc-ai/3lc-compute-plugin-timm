# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""The native timm form and MCP catalog use the same job configuration."""

import sys
from pathlib import Path

from tlc_plugin_sdk.job_contract import JobContract, form_field_schema, write_tools

from tlc_plugin_timm.form_fields import PARAM_FIELDS

# Feature extraction can operate on unlabelled images. Classification cannot.
_TASK_REQUIREMENTS = {
    "if": {"properties": {"task_type": {"const": "feature_extraction"}}, "required": ["task_type"]},
    "else": {"properties": {"label_column": {"minLength": 1}}, "required": ["label_column"]},
}


class TimmContract(JobContract):
    def native_schema(self):
        schema = super().native_schema()
        schema["allOf"] = [_TASK_REQUIREMENTS]
        return schema

    def tool(self, *args, **kwargs):
        tool = super().tool(*args, **kwargs)
        tool["input_schema"]["allOf"] = [_TASK_REQUIREMENTS]
        return tool


def contract():
    fields = {"params." + field["id"]: form_field_schema(field) for field in PARAM_FIELDS}
    fields.update({
        "model_name": {"type": "string", "minLength": 1},
        "run_name": {"type": "string", "default": ""},
        "project_name": {"type": "string", "default": ""},
        "task_type": {"type": "string", "enum": ["", "classification", "feature_extraction"], "default": ""},
        "train_table_url": {"type": "string", "minLength": 1},
        "val_table_url": {"type": "string", "default": ""},
        "image_column": {"type": "string", "minLength": 1},
        "label_column": {"type": "string", "default": ""},
        "use_latest": {"type": "boolean", "default": False},
        "mode": {"type": "string", "enum": ["train", "collect"], "default": "train"},
        "params.device": {"type": "string", "enum": ["", "auto", "cpu", "cuda"], "default": "auto"},
        "params.pretrained_model_url": {"type": "string", "default": ""},
        "params.copy_model_to_run": {"type": "boolean", "default": True},
    })
    return TimmContract(
        fields,
        required=("train_table_url", "model_name", "image_column"),
        aliases={"model_name": "model"},
        metadata={key: {"type": "string"} for key in ("id", "name", "created")}
        | {"last_run": {"type": ["string", "null"]}},
    )


def tools():
    native = contract()
    return [
        native.tool(
            "train",
            "Train an image classifier",
            "Train with the native timm form settings, including augmentation, metrics and embeddings. "
            "For remote data use plan_data/prepare_data; follow job_status for the run_url.",
            constants={"mode": "train"},
        ),
        native.tool(
            "collect",
            "Collect classifier predictions and embeddings",
            "Use the native timm collection mode without training. Select the model and optional saved weights. "
            "Uses the same data preparation and job queue as the module.",
            constants={"mode": "collect"},
        ),
        {
            "name": "models",
            "title": "Find timm models",
            "description": "Find available timm model names by family.",
            "input_schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "q": {"type": "string", "minLength": 1, "maxLength": 80},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 30},
                },
                "required": ["q", "limit"],
            },
            "annotations": {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True},
            "route": {"method": "GET", "path": "/models", "query": ["q", "limit"]},
        },
    ]


if __name__ == "__main__":
    write_tools(Path(__file__).with_name("mcp-tools.json"), tools(), check="--check" in sys.argv)
