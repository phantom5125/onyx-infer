"""Prepare H3's ordered vision features and reference latent patches."""

import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image, ImageOps
from tokenizers import Tokenizer

from onyx.reference.vae_encoder import encode_reference
from onyx.reference.vision import visual_features
from onyx.reference.weights import Weights


def prepare(models, work, prompt_ids, paths, seed):
    tokenizer = Tokenizer.from_file(str(models / "tokenizer/tokenizer.json"))
    ids, positions, tags, visual_rows, grids = [], [], [], [], []
    features = {name: [] for name in ("vision", "deepstack_0", "deepstack_1", "deepstack_2")}
    patches = []
    position = 0

    def append_text(tokens, tag=1):
        nonlocal position
        ids.extend(tokens)
        positions.extend((offset, offset, offset) for offset in range(position, position + len(tokens)))
        tags.extend([tag] * len(tokens))
        position += len(tokens)

    latent_seed, noise_seed = np.random.SeedSequence(seed).spawn(2)
    latent_rng = np.random.default_rng(latent_seed)
    noise_rng = np.random.default_rng(noise_seed)
    vision = Weights(models / "text_encoder.onxpack", resident=True)
    vae = Weights(models / "vae/diffusion_pytorch_model.safetensors", resident=True)
    stats = json.loads((models / "vae/config.json").read_text())
    try:
        for number, path in enumerate(paths, 1):
            with Image.open(path) as image:
                short, long = sorted(image.size)
                if short < 128 or long > 8192 or long > short * 4:
                    raise ValueError(f"Reference {number} must be 128–8192 pixels per side, up to 4:1")
                rgb = ImageOps.exif_transpose(image).convert("RGB")
            append_text(tokenizer.encode(f"Picture {number}: ", add_special_tokens=False).ids)
            append_text([151652], tag=0)
            merged, deepstack, (rows, columns) = visual_features(vision, rgb)
            count = rows * columns
            visual_rows.extend(range(len(ids), len(ids) + count))
            ids.extend([151655] * count)
            positions.extend((position, position + y, position + x)
                             for y in range(rows) for x in range(columns))
            tags.extend([0] * count)
            position += max(rows, columns)
            append_text([151653], tag=0)
            for key, value in zip(features, (merged, *deepstack), strict=True):
                value = value.astype(mx.bfloat16)
                mx.eval(value)
                features[key].append(value)
            latent = np.asarray(encode_reference(vae, rgb, stats, latent_rng))
            channels, height, width = latent.shape
            rows, columns = height // 2, width // 2
            grids.append((rows, columns))
            latent = 0.999 * latent + 0.001 * noise_rng.standard_normal(latent.shape, dtype=np.float32)
            packed = latent.reshape(channels, rows, 2, columns, 2).transpose(1, 3, 0, 2, 4)
            patches.append(mx.array(packed.reshape(rows * columns, channels * 4)))
            print(f"Encoded reference {number}/{len(paths)}", file=sys.stderr, flush=True)
    finally:
        vision.close()
        vae.close()
    append_text(prompt_ids)
    (work / "tokens.json").write_text(json.dumps({"ids": ids, "count": len(ids)}))
    mx.save_safetensors(str(work / "vision.safetensors"), {
        "positions": mx.array(positions, dtype=mx.int32),
        "rows": mx.array(visual_rows, dtype=mx.int32),
        **{key: mx.concatenate(values, axis=0) for key, values in features.items()},
    })
    mx.save_safetensors(str(work / "references.safetensors"), {
        "text_tags": mx.array(tags, dtype=mx.int32),
        "grids": mx.array(grids, dtype=mx.int32),
        "patches": mx.concatenate(patches, axis=0),
    })


if __name__ == "__main__":
    directory = Path(sys.argv[1])
    request = json.loads((directory / "reference-request.json").read_text())
    prepare(Path(request["models"]), directory, request["prompt_ids"], request["paths"], request["seed"])
