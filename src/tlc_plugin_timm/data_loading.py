# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Carry the job's resolved image locations into spawned data readers."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial


def _initialize_reader(worker_id: int, *, aliases: dict[str, str]) -> None:
    import tlc

    # Tables may already be fully defined when pickled. Loading them in a new
    # process therefore need not reload project aliases. Preserve the parent's
    # effective aliases, including host-authorized staging overrides.
    for token, path in aliases.items():
        tlc.url.register_url_alias(token=token, path=path, force=True)


def reader_initializer() -> Callable[[int], None]:
    import tlc

    aliases = {str(token): str(path) for token, path in tlc.url.get_registered_url_aliases().items()}
    return partial(_initialize_reader, aliases=aliases)
