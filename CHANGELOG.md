# Changelog

All notable changes to `3lc-compute-plugin-timm` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Nothing yet.

## [0.2.0] - 2026-08-19

### Added
- The pretrained-checkpoint field uses the SDK's shared data-source picker: browse the compute
  node's filesystem (confined to operator-configured roots) instead of typing a path blind.
  The SDK's `/browse` route is mounted alongside the plugin's own routes (#5).

### Changed
- **Distribution moved to PyPI**: tagged releases publish `3lc-compute-plugin-timm` to public
  PyPI via Trusted Publishing; the CloudRepo index (pypi.3lc.ai) is no longer needed to install
  the plugin. Manual prerelease builds keep publishing to CloudRepo for a grace period (#5).
- The plugin SDK pin is `>=0.2.2,<0.3.0`, resolved from public PyPI (the SDK's home since
  0.2.2) — no custom indexes remain besides the CUDA torch index (#5). Earlier steps on the
  way: the pin was widened to `>=0.2.0,<0.3.0` (#2), and `3lc` moved to public PyPI with the
  3.2 rust release (#3).

### Fixed
- Training jobs are attributed to the configured 3LC project in the generic Queue &
  Progress panel: the run request now carries `project_name`, which the host reads from
  the request body only — the value stored in the saved config never reached the job
  record, so jobs were hidden from every project-filtered view (#4).
- A `~`-prefixed pretrained-checkpoint path is expanded at ingress instead of reaching
  `torch.load` literally and failing mid-job with an opaque file-not-found (#5).

## [0.1.3] - 2026-07-03

### Fixed
- The plugin manifest version and the distribution version are bumped together, so the version
  the plugin card reports matches the installed distribution.

## [0.1.2] - 2026-07-03

### Fixed
- The CUDA torch index is applied on Windows as well as Linux, so GPU-enabled installs work on
  Windows hosts.

### Changed
- The plugin SDK dependency is resolved from the public package index under its final name
  `3lc-compute-plugin-sdk` (was a git pin).

## [0.1.1] - 2026-07-01

### Added
- PaCMAP and UMAP dimensionality-reduction dependencies (via the `3lc[pacmap,umap]` extras) so
  embedding visualizations work out of the box.

## [0.1.0] - 2026-07-01

First release, extracted from the `3lc-compute-plugins` umbrella into its own repository.

### Added
- The timm training plugin for the 3LC compute service: fine-tune any of the 1000+ PyTorch Image
  Models on 3LC tables, with per-sample metrics and embeddings collected to 3LC runs. GPU-classed
  and venv-isolated; distributed as `3lc-compute-plugin-timm`.
