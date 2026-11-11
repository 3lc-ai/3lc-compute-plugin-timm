# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Authoritative timm controls, constraints and defaults; no ML imports."""

PARAM_FIELDS = [
    {"id": "epochs", "type": "number", "default": 20, "min": 1, "max": 1000, "step": 1},
    {"id": "batch_size", "type": "number", "default": 32, "min": 1, "max": 512, "step": 1},
    {"id": "num_workers", "type": "number", "default": 8, "min": 0, "max": 32, "step": 1},
    {"id": "image_size", "type": "number", "default": 224, "min": 32, "max": 1024, "step": 32},
    {"id": "lr", "type": "number", "default": 0.001, "min": 1e-05, "max": 1, "step": 0.0001},
    {"id": "weight_decay", "type": "number", "default": 0.0001, "min": 0, "max": 1, "step": 0.0001},
    {
        "id": "optimizer",
        "type": "select",
        "options": [
            {"value": "adamw", "label": "AdamW"},
            {"value": "adam", "label": "Adam"},
            {"value": "sgd", "label": "SGD"},
            {"value": "lamb", "label": "LAMB"},
        ],
        "default": "adamw",
    },
    {
        "id": "scheduler",
        "type": "select",
        "options": [
            {"value": "cosine", "label": "Cosine"},
            {"value": "step", "label": "Step"},
            {"value": "plateau", "label": "Plateau"},
        ],
        "default": "cosine",
    },
    {"id": "warmup_epochs", "type": "number", "default": 3, "min": 0, "max": 50, "step": 1},
    {
        "id": "auto_augment",
        "type": "select",
        "options": [
            {
                "value": "",
                "label": "None",
                "title": "No automatic augmentation — only basic transforms (resize, flip, jitter)",
            },
            {
                "value": "rand-m9-mstd0.5",
                "label": "RandAugment (m9)",
                "title": (
                    "Randomly applies 2 of 14 transforms (rotate, "
                    "shear, translate, etc.) at moderate intensity (magnitude 9/30)"
                ),
            },
            {
                "value": "rand-m15-mstd0.5",
                "label": "RandAugment (m15)",
                "title": (
                    "Same as RandAugment but stronger intensity (magnitude 15/30) "
                    "— better for larger datasets and longer training"
                ),
            },
            {
                "value": "trivialaugment",
                "label": "TrivialAugment",
                "title": (
                    "Applies a single random transform at random intensity per image "
                    "— simpler than RandAugment, no tuning needed, often equally effective"
                ),
            },
        ],
        "default": "rand-m9-mstd0.5",
    },
    {"id": "reprob", "type": "number", "default": 0, "min": 0, "max": 1, "step": 0.05},
    {"id": "mixup_alpha", "type": "number", "default": 0, "min": 0, "max": 1, "step": 0.1},
    {"id": "cutmix_alpha", "type": "number", "default": 0, "min": 0, "max": 1, "step": 0.1},
    {"id": "hflip", "type": "checkbox", "default": True},
    {"id": "color_jitter", "type": "checkbox", "default": True},
    {
        "id": "image_embeddings_dim",
        "type": "select",
        "options": [{"value": 0, "label": "Disabled"}, {"value": 2, "label": "2D"}, {"value": 3, "label": "3D"}],
        "default": 2,
    },
    {
        "id": "image_embeddings_reducer",
        "type": "select",
        "options": [{"value": "umap", "label": "UMAP"}, {"value": "pacmap", "label": "PaCMAP"}],
        "default": "umap",
    },
    {"id": "write_full_embeddings", "type": "checkbox", "default": False},
    {"id": "collection_epoch_start", "type": "number", "default": "", "min": 1, "max": 1000, "step": 1},
    {"id": "collection_interval", "type": "number", "default": 1, "min": 1, "max": 100, "step": 1},
    {"id": "collect_loss", "type": "checkbox", "default": True},
    {"id": "sampling_weights", "type": "checkbox", "default": False},
    {"id": "exclude_zero_weight_training", "type": "checkbox", "default": False},
    {"id": "collection_val_only", "type": "checkbox", "default": False},
    {"id": "collection_disable", "type": "checkbox", "default": False},
]
