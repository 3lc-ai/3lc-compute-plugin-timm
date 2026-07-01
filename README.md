# 3lc-plugin-timm

The **timm training** plugin for the [3LC compute service](https://github.com/3lc-ai) — fine-tune
any of 1,000+ pretrained image classifiers from [timm](https://github.com/huggingface/pytorch-image-models)
(PyTorch Image Models) on your data, with per-sample metrics, embedding collection, and live
SocketIO progress.

A standalone, venv-isolated plugin distribution, licensed **Apache-2.0**.

## How it's consumed

The host never installs this distribution into its own venv. It is delivered through any of the
three plugin Sources, all converging on the same out-of-process worker in a managed venv:

- **Folder Source (dev):** point the service at this repo's `src/`
  (`--plugin-dir ../3lc-plugin-timm/src` or `TLC_COMPUTE_EXTERNAL_PLUGIN_DIRS`). Provisioning runs
  `uv sync --extra timm` against this repo.
- **Index:** `3lc-plugin-timm[timm]==<ver>`.
- **GitHub:** `github:3lc-ai/3lc-plugin-timm@v<ver>`.

The heavy stack (`timm`, `torch`, `torchvision`, `pacmap`, `umap-learn`) lives behind the **`[timm]`
extra** named by `runtime.provision_extra` in `src/tlc_plugin_timm/plugin.toml` and is installed
**only** into the plugin's provisioned venv — never the host venv. The base dependency is the SDK
floor only.

## Dev setup

```bash
uv sync --extra timm     # exactly what the host provisions into the plugin's venv
uvx --from 'ruff>=0.15,<0.16' ruff check .
```

To develop against a sibling `3lc-plugin-sdk` checkout, override its source **uncommitted**:

```toml
# pyproject.toml [tool.uv.sources]  (local dev only — do not commit)
3lc-plugin-sdk = { path = "../3lc-plugin-sdk", editable = true }
```

The plugin contract and author guide live in
[`3lc-plugin-sdk`](https://3lc-ai.github.io/3lc-plugin-sdk/).
