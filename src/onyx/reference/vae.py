import json
import math

import mlx.core as mx
import numpy as np
from PIL import Image

from onyx.reference.weights import Weights, attention, norm, rope, swiglu


def tile_positions(size):
    if size <= 16:
        return [0]
    count = math.ceil((size - 4) / 12)
    extra = 16 * count - 4 * (count - 1) - size
    gaps = [12 - extra // (count - 1) - (index < extra % (count - 1)) for index in range(count - 1)]
    return [0, *np.cumsum(gaps).tolist()]


def decode_tiles(weights, z):
    batch, height, width, channels = z.shape
    kind = z.dtype
    x = z.reshape(batch, -1, channels)
    post = weights["post_quant_conv.weight"].reshape(channels, channels).astype(kind)
    x = x @ post.T + weights["post_quant_conv.bias"].astype(kind)
    x = weights.linear("decoder.x_embedder", x)
    dim = weights.header["decoder.x_embedder.weight"]["shape"][0]
    x = mx.concatenate([
        x,
        mx.broadcast_to(weights["decoder.register_tokens"].astype(kind), (batch, 4, dim)),
        mx.zeros((batch, 1, dim), dtype=kind),
    ], axis=1)
    coordinates = np.meshgrid(
        (np.arange(height) + 0.5) * 2 / height - 1,
        (np.arange(width) + 0.5) * 2 / width - 1,
        indexing="ij",
    )
    positions = np.stack(coordinates, axis=-1).reshape(-1, 2)
    positions = np.concatenate([positions, np.zeros((5, 2))])
    angles_np = (2 * np.pi * positions[:, :, None] * 100 ** (-np.arange(8) / 8)).reshape(len(positions), -1)
    angles = mx.array(angles_np.astype(np.float32))
    layers = sum(key.endswith(".attn.to_qkv.weight") for key in weights.header)
    for layer in range(layers):
        prefix = f"decoder.transformer_blocks.{layer}"
        qkv = weights.linear(prefix + ".attn.to_qkv", norm(x, weights[prefix + ".norm1.weight"]))
        qkv = qkv.reshape(batch, x.shape[1], dim // 64, 192).transpose(0, 2, 1, 3)
        q, k, v = mx.split(qkv, 3, axis=-1)
        q, k = rope(norm(q), angles), rope(norm(k), angles)
        out = attention(q, k, v, scale=0.125)
        x = x + weights.linear(prefix + ".attn.to_out", out.reshape(x.shape)) * weights[prefix + ".scale1"].astype(kind)
        out = swiglu(weights.linear(prefix + ".ff.w1", norm(x, weights[prefix + ".norm2.weight"])))
        x = x + weights.linear(prefix + ".ff.w2", out) * weights[prefix + ".scale2"].astype(kind)
        mx.eval(x)
    x = mx.fast.layer_norm(
        x[:, :height * width],
        weights["decoder.norm_out.weight"].astype(kind),
        weights["decoder.norm_out.bias"].astype(kind),
        1e-5,
    )
    out = weights.linear("decoder.proj_out", x)
    return out.reshape(batch, height, width, 3, 16, 16).transpose(0, 1, 4, 2, 5, 3).reshape(batch, height * 16, width * 16, 3)


def decode(weights_path, latent, config_path, output_path):
    config = json.loads(config_path.read_text())
    weights = Weights(weights_path, resident=True)
    try:
        source_dtype = weights.header["decoder.x_embedder.weight"]["dtype"]
        kind = mx.bfloat16 if source_dtype == "BF16" else mx.float16 if source_dtype == "F16" else mx.float32
        z = latent.transpose(1, 2, 0) * mx.array(config["latents_std"]) + mx.array(config["latents_mean"])
        z = z.astype(kind)
        ys, xs = tile_positions(z.shape[0]), tile_positions(z.shape[1])
        canvas = np.zeros((z.shape[0] * 16, z.shape[1] * 16, 3), dtype=np.float32)
        for row, y in enumerate(ys):
            strip = np.zeros((256, canvas.shape[1], 3), dtype=np.float32)
            for start in range(0, len(xs), 4):
                columns = xs[start:start + 4]
                group = mx.stack([z[y:y + 16, x:x + 16] for x in columns])
                tiles = np.asarray(decode_tiles(weights, group).astype(mx.float32))
                for offset, (tile, x) in enumerate(zip(tiles, columns)):
                    column = start + offset
                    tile = tile.copy()
                    if row:
                        overlap = (ys[row - 1] + 16 - y) * 16
                        alpha = np.arange(overlap, dtype=np.float32)[:, None, None] / overlap
                        old = canvas[16 * y:16 * y + overlap, 16 * x:16 * (x + 16)]
                        tile[:overlap] = old + (tile[:overlap] - old) * alpha
                    if column:
                        overlap = (xs[column - 1] + 16 - x) * 16
                        alpha = np.arange(overlap, dtype=np.float32)[None, :, None] / overlap
                        old = strip[:, 16 * x:16 * x + overlap]
                        tile[:, :overlap] = old + (tile[:, :overlap] - old) * alpha
                    strip[:, 16 * x:16 * (x + 16)] = tile
            canvas[16 * y:16 * (y + 16)] = strip
        rgb = canvas * np.array([0.229, 0.224, 0.225], np.float32) + np.array([0.485, 0.456, 0.406], np.float32)
        if not np.isfinite(rgb).all():
            raise ArithmeticError("Non-finite VAE output")
        Image.fromarray((rgb.clip(0, 1) * 255).round().astype(np.uint8)).save(output_path)
    finally:
        weights.close()
