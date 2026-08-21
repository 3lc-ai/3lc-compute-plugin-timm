# 3lc-compute-plugin-timm

A [3LC Hub](https://docs.3lc.ai) compute-service plugin for fine-tuning image classifiers.
Train any of the 1,000+ pretrained backbones from
[timm](https://github.com/huggingface/pytorch-image-models) (PyTorch Image Models) on your data,
with per-sample metrics, embedding collection, and live progress in the Hub.

## How it's used

You don't install this yourself. The 3LC Hub provisions the plugin into its own isolated
environment (including the GPU stack) and runs it for you; it then appears in the Hub next to the
built-in tools.

## License

Apache-2.0. See `LICENSE`.

## Links

- 3LC Hub documentation: <https://docs.3lc.ai>
- Plugin SDK & author guide: <https://3lc-ai.github.io/3lc-compute-plugin-sdk/>
