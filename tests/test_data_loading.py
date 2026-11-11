# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Spawned data readers must retain project aliases and staged overrides."""

from __future__ import annotations

import pytest


def test_spawned_reader_resolves_project_image(tmp_path):
    torch = pytest.importorskip("torch")
    tlc = pytest.importorskip("tlc")
    from PIL import Image
    from torchvision.transforms import ToTensor

    from tlc_plugin_timm.data_loading import reader_initializer
    from tlc_plugin_timm.trainer import _SampleTransform

    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (8, 8), (255, 0, 0)).save(images / "sample.png")
    token = "TIMM_READER_TEST"
    tlc.url.register_url_alias(token, str(images), force=True)
    try:
        writer = tlc.TableWriter(
            project_name="reader-test",
            dataset_name="images",
            table_name="initial",
            root_url=str(tmp_path / "projects"),
            schema={"image": tlc.schemas.ImageSchema(sample_type="url"), "label": tlc.schemas.Int64Schema()},
        )
        writer.add_row({"image": f"<{token}>/sample.png", "label": 1})
        table = writer.finalize()
        assert table.get_column_as_pyarrow_array("image")[0].as_py().startswith(f"<{token}>/")
        # Match training: materialize before sending the table to child readers.
        table[0]
        loader = torch.utils.data.DataLoader(
            table.with_transform(_SampleTransform(ToTensor(), "image", "label")),
            batch_size=1,
            num_workers=1,
            multiprocessing_context="spawn",
            worker_init_fn=reader_initializer(),
        )
        batch, labels = next(iter(loader))
        assert tuple(batch.shape) == (1, 3, 8, 8)
        assert batch[0, 0, 0, 0].item() == 1
        assert labels.tolist() == [1]
    finally:
        tlc.url.unregister_url_alias(token)
