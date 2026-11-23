# Hardware and local validation

Use `tools/native_validate.py` locally or in GitHub Actions to produce `onyx.ci.v1` JSON and Markdown reports.

| Suite | Checks | H3 model pack |
| --- | --- | --- |
| `contract` | Locked Python dependencies, schema compatibility, tests, CMake build, and Metal CTest | Optional |
| `metal` | Metal CTest and Q4 GEMM | Optional |
| `profiles` | Profile registry and matching matrices, with historical matrices identified separately | Optional |
| `kernel` | Q4 GEMM for a selected variant, shape, and group size | Optional |
| `h3` | Native generation and replay | Required |
| `h3-e2e` | Native generation and community comparison | Required, including community weights for full reproduction |
| `all` | Contract, Metal, and native generation | Required |

Choose `teapot-1024`, `teapot-4096`, or a request JSON via `--case`. Use `metal-only` for a run without image generation. The archived VAE matrix records measurements from an earlier profile registry. Bind a new matrix to the registry SHA-256 used for its run.

Use `--skip-build` to reuse the current build. Use `-DONYX_ENABLE_MLX=OFF` to validate a build without MLX. For a standalone Metal compiler set `METAL_COMPILER`. To enable real-weight CTest probes pass `-DONYX_Q4_TEST_PACK=/path/to/probe.onxpack`.

## Hosted and self-hosted workflows

The hosted `CI` workflow checks schema compatibility, Python, the CMake build, Metal probes, Q4 GEMM, and profile evidence. `Native validation` runs via manual dispatch or its weekly schedule on a runner labeled `self-hosted`, `macOS`, `ARM64`, and `onyx-metal`.

The reference runner is an Apple Silicon Mac Studio with M5 Max and 36 GB of unified memory, macOS 27, Metal 4, Apple Clang 21, and Metal compiler `32023.921`. Stage and verify model artifacts before running hardware validation. A maintainer reviews contributions before dispatching self-hosted validation at the intended revision.

## Model preparation

Follow the [first-image guide](QUICKSTART.md) to download and convert the pinned checkpoint.

The native suites look under `artifacts/models/h3-w4` for these files.

- `manifest.pb`
- `transformer.onxpack` and `text_encoder.onxpack`
- `pdd_fused_heads.safetensors` and `pdd_config.json`
- `vae/diffusion_pytorch_model.safetensors` and `vae/config.json`
- `tokenizer/tokenizer.json`

A report marked `blocked` lists missing prerequisites. A report marked `fail` records the failed command or acceptance check.

## Examples

```bash
python tools/native_validate.py --suite contract
python tools/native_validate.py --suite metal
python tools/native_validate.py \
  --suite kernel --variant tile32 \
  --m 1024 --n 21504 --k 5376 --group 64 --iterations 10 \
  --label local-q4-tile32
python tools/native_validate.py --suite h3 --case teapot-1024 --runs 3 --label local-native-1024
python tools/native_validate.py --suite h3 --case teapot-4096 --label local-native-4096
python tools/native_validate.py --suite all --runs 1 --label local-full
```

Keep the GPU and machine available for the measurement. Give each invocation a unique `--label`, such as `pr-123-q4-tile32`.

## Reports

Reports under `artifacts/ci/` include the source revision, dirty state, target, commands, return codes, durations, and log tails. Kernel reports also include correctness, shapes, and wall-time and GPU medians. Native reports include latency, peak memory, and output hash consistency. Per-run records and PNGs live at `artifacts/benchmarks/onyx-native-h3/<case-id>/<run-id>/`.

Attach the JSON and Markdown files to pull requests or use the workflow artifacts. Include the measured shapes, hardware, output hashes, and any cases still to be verified.

For every precision-sensitive change, also upload the original baseline and candidate PNGs directly to the pull request so reviewers can inspect them without running the model. Keep the request, model, seed, resolution, sampling steps, and execution profile fixed; record both revisions and output hashes beside the images. A hash proves identity, not visual quality, so it does not replace the image evidence. See [Generated-image evidence](../CONTRIBUTING.md#generated-image-evidence).
