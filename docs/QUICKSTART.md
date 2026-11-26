# Generate your first image

You need an Apple Silicon Mac with macOS 26.2 or later, the Xcode Metal 4 toolchain, and [uv](https://docs.astral.sh/uv/getting-started/installation/) with Python 3.12 or later. The 1024×1024 workflow has been tested on 24 GB and 36 GB Macs. The minimum unified memory requirement is unverified.

Allow about 200 GB of disk space for the 135 GB source checkpoint and converted files together, plus space for build tools. The W4 inference files total about 62 GB including the VAE, PDD heads, and tokenizer. The conversion symlinks several files from the source checkpoint, so keep the linked files in place.

Review the [MiniMax-H3-Image license at the pinned ModelScope revision](https://modelscope.cn/models/MiniMax/MiniMax-H3-Image/resolve/df334556f585a596a710cf4f7ce3a567c0738ee5/LICENSE) before obtaining the model. It defines geographic and use restrictions. The Onyx Infer release provides source code and does not bundle model weights.

## Build the native runtime

```bash
git clone https://github.com/phantom5125/onyx-infer.git
cd onyx-infer
uv sync --frozen --no-dev

xcrun --find metal
uvx --from cmake cmake -S . -B .build -DCMAKE_BUILD_TYPE=Release -DONYX_ENABLE_MLX=OFF
uvx --from cmake cmake --build .build --target onyx-native --parallel 8
```

If `xcrun --find metal` fails, select a full Xcode installation and install its Metal toolchain. If you have a standalone Metal toolchain, pass its compiler path as `-DMETAL_COMPILER=/path/to/metal` at configure time.

## Prepare the model

Run these commands from the repository root. The download script resumes interrupted transfers and verifies the SHA-256 values in the checked-in source lock. The two W4 conversions can take considerable time.

```bash
source_dir=artifacts/models/h3-image-source
meta_dir=artifacts/models/h3-original-meta
model_dir=artifacts/models/h3-w4

uv run --frozen --no-dev python tools/modelscope_fetch.py \
  --source-lock config/sources/h3-image-files.json --directory "$source_dir"

uv run --frozen --no-dev onyx import \
  --source "$source_dir" --lock config/sources/h3-image-files.json --output "$meta_dir"

uv run --frozen --no-dev onyx quantize --source "$source_dir" \
  --component text_encoder --all-linear --output "$model_dir/text_encoder.onxpack"
uv run --frozen --no-dev onyx quantize --source "$source_dir" \
  --component transformer --all-linear --output "$model_dir/transformer.onxpack"

uv run --frozen --no-dev onyx package-w4 --source "$source_dir" \
  --original-manifest "$meta_dir/manifest.pb" --output "$model_dir"
```

## Generate a 1024×1024 image

```bash
uv run --frozen --no-dev onyx native-generate \
  --models artifacts/models/h3-w4 \
  --prompt "A yellow enamel teapot on a blue wooden table." \
  --width 1024 --height 1024 --seed 42 \
  --output artifacts/native-output.png

open artifacts/native-output.png
```

See the [hardware validation guide](VALIDATION.md) for test commands and the [native tuning guide](NATIVE_TUNING.md) for optional caches and larger images.

## Edit with reference images

Build `onyx-native` from the current source and install the optional Python `reference` extra. The extra uses MLX to encode the reference pixels. Onyx runs the language encoder, DiT and VAE decoder with the existing H3 W4 package.

```bash
uv sync --frozen --no-dev --extra reference
uv run --frozen --no-dev --extra reference onyx native-generate \
  --models artifacts/models/h3-w4 \
  --reference room.png --reference chair.png \
  --prompt "Add the chair from <Picture 2> to the room in <Picture 1>." \
  --width 1024 --height 1024 --seed 42 \
  --output artifacts/native-edit.png
```

Pass `--reference` once per image, up to 16 times. `<Picture 1>` refers to the first argument. Each image must have sides between 128 and 8192 pixels and an aspect ratio of at most 4:1. Larger reference sets and output sizes increase the number of tokens and the memory required. The 16-reference path was verified with square inputs and 1024×1024 output on an M5 Max with 36 GB of unified memory. Reference edits are exposed through the CLI. [16-reference validation](../benchmarks/results/m5-max-36gb-h3-ref16-adaln-v1.json)

To try the experimental DiT memory optimization, prefix the generation command with `ONYX_DIT_INDEXED_MODULATION=1`. It uses a compact modulation table for reference edits. The reported memory figures come from the verified 16-reference case.

For a replay case, record the inputs in `request.references` in order, with each file's byte size, SHA-256 and local path. A relative path is resolved from the case file's directory. Include a standalone `{references}` argument in the generator command to pass the verified files to `onyx native-generate`.
