import json

import mlx.core as mx
import numpy as np

from onyx.reference.weights import Weights, attention, heads, norm, rope, swiglu


def reorder_original_qkv(x):
    rows = x.shape[0]
    return x.reshape(rows, 56, 3, 128).transpose(0, 2, 1, 3).reshape(rows, 56 * 3 * 128)


def qkv_attention(weights, prefix, x, angles=None, original=False):
    fused = weights.linear(prefix + ".qkv_proj", x)
    if original:
        fused = reorder_original_qkv(fused)
    q, k, v = (heads(value) for value in mx.split(fused, 3, axis=-1))
    q = norm(q, weights[prefix + ".q_norm.weight"])
    k = norm(k, weights[prefix + ".k_norm.weight"])
    if angles is not None:
        q, k = rope(q, angles), rope(k, angles)
    return weights.linear(prefix + ".out_proj", attention(q, k, v))


def mlp(weights, prefix, x):
    return weights.linear(prefix + ".fc2", swiglu(weights.linear(prefix + ".fc1", x)))


def prepare_text(weights, embedding, original=False):
    x = weights.linear("condition_proj", embedding.astype(mx.bfloat16))
    for layer in range(2):
        prefix = f"token_refiner.blocks.{layer}"
        x = x + qkv_attention(weights, prefix + ".attn", norm(x, weights[prefix + ".norm1.weight"]), original=original)
        x = x + mlp(weights, prefix + ".mlp", norm(x, weights[prefix + ".norm2.weight"]))
    return norm(x, weights["token_refiner.final_norm.weight"])


def curve_coordinates(weights, sigmas):
    t = np.clip(1 - np.asarray(sigmas, dtype=np.float32), 0, 1)
    table = np.asarray(weights["adaln_t_table"])
    position = t * (table.shape[0] - 1)
    index = np.minimum(position.astype(np.int32), len(table) - 2)
    return table[index] + (table[index + 1] - table[index]) * (position - index.astype(np.float32))[:, None]


def precompute(weights, prefix, coordinates, modalities, parts):
    projection = prefix + ".adaln_proj.linear"
    weight = weights[projection + ".weight"].astype(mx.float32)
    bias = weights[projection + ".bias"].astype(mx.float32)
    values = mx.array(coordinates) @ weight.T + bias
    return values.reshape(len(coordinates), modalities, parts, -1)


def timestep_projection(timesteps):
    half = 128
    exponent = -np.log(10000.0) * np.arange(half, dtype=np.float32) / half
    angles = np.asarray(timesteps, dtype=np.float32)[:, None] * np.exp(exponent)[None]
    return mx.array(np.concatenate([np.cos(angles), np.sin(angles)], axis=1))


def original_time_embedding(weights, sigmas):
    times = 1 - np.asarray(sigmas[:-1], dtype=np.float32)
    x = weights.linear("time_embedder.proj_in", timestep_projection(times))
    x = x * mx.sigmoid(x)
    return weights.linear("time_embedder.proj_out", x)


def original_precompute(weights, prefix, temb, final=False):
    projection = prefix + ".adaln_proj.linear"
    dtype = weights[projection + ".weight"].dtype
    values = weights.linear(projection, (temb * mx.sigmoid(temb)).astype(dtype))
    if final:
        return values.reshape(temb.shape[0], 1, 2, -1)
    return values.reshape(temb.shape[0], 3, 6, -1)


def modulate(x, table, segments, scale=False):
    return mx.concatenate([
        x[start:stop] * (1 + table[row].astype(x.dtype) if scale else table[row].astype(x.dtype))
        for start, stop, row in segments
    ])


def shifted(x, scale, shift, segments):
    return mx.concatenate([
        x[start:stop] * (1 + scale[row].astype(x.dtype)) + shift[row].astype(x.dtype)
        for start, stop, row in segments
    ])


def image_positions(rows, columns, cursor):
    area = (rows * columns) ** 0.5

    def axis(count):
        return (np.arange(count) * (count / area / count) + (1 - count / area) / 2) * 32

    yy, xx = np.meshgrid(axis(rows), axis(columns), indexing="ij")
    return np.stack([np.full(rows * columns, cursor), yy.flatten(), xx.flatten()], axis=-1)


def rotary_positions(text_count, rows, columns, frequencies):
    text = np.zeros((text_count, 3), dtype=np.float32)
    text[:, 0] = np.arange(text_count)
    positions = np.concatenate([text, image_positions(rows, columns, text_count)])
    return mx.array((positions[:, :, None] * np.asarray(frequencies)).reshape(len(positions), -1).astype(np.float32))


def patchify(latent):
    channels, height, width = latent.shape
    return latent.reshape(channels, height // 2, 2, width // 2, 2).transpose(1, 3, 0, 2, 4).reshape(-1, channels * 4)


def denoise(weights_path, embedding, width, height, seed, resident=True, original=False, pdd_path=None, sigmas=None, raw_source=None):
    weights = Weights(weights_path, resident=resident, raw_source=raw_source)
    pdd = Weights(pdd_path) if pdd_path else None
    try:
        if sigmas is None:
            metadata = weights.header.get("__metadata__") or {}
            sigmas = json.loads(metadata["config"])["transformer"]["pdd_sigmas"]
        if len(sigmas) != 7:
            raise ValueError("The reference path expects the six-step H3 schedule")
        prefixes = [f"blocks.{index}" for index in range(50)]
        if original:
            if not pdd:
                raise ValueError("Original H3 requires the separate PDD heads")
            temb = original_time_embedding(weights, sigmas)
            modulations = [original_precompute(weights, prefix, temb) for prefix in prefixes]
            final_modulation = original_precompute(weights, "final_layer", temb, final=True)
        else:
            coordinates = curve_coordinates(weights, sigmas[:-1])
            modulations = [precompute(weights, prefix, coordinates, 3, 6) for prefix in prefixes]
            final_modulation = precompute(weights, "final_layer", coordinates, 1, 2)
        text = prepare_text(weights, embedding, original=original)
        mx.eval(text, final_modulation, *modulations)
        weights.release()
        mx.clear_cache()
        text_count = text.shape[0]
        rows, columns = height // 32, width // 32
        angles = rotary_positions(text_count, rows, columns, weights["rope.inv_freq"])
        rng = np.random.default_rng(seed)
        latent = mx.array(rng.standard_normal((24, rows * 2, columns * 2), dtype=np.float32))
        patches = mx.array(patchify(np.asarray(latent)))
        segments = [(0, text_count, 1), (text_count, text_count + rows * columns, 0)]
        for step, (sigma, next_sigma) in enumerate(zip(sigmas, sigmas[1:])):
            image = weights.linear("video_patch_proj", patches)
            x = mx.concatenate([text, image.astype(text.dtype)])
            for layer, prefix in enumerate(prefixes):
                table = modulations[layer][step]
                h = shifted(norm(x, weights[prefix + ".norm1.weight"]), table[:, 1], table[:, 0], segments)
                x = x + modulate(qkv_attention(weights, prefix + ".attn", h, angles, original=original), table[:, 2], segments)
                h = shifted(norm(x, weights[prefix + ".norm2.weight"]), table[:, 4], table[:, 3], segments)
                x = x + modulate(mlp(weights, prefix + ".mlp", h), table[:, 5], segments)
                mx.eval(x)
            shift, scale = final_modulation[step, 0]
            target = norm(x[text_count:], weights["final_layer.norm.weight"])
            if original:
                target = (target * (1 + scale) + shift).astype(mx.float32)
            else:
                target = target.astype(mx.float32) * (1 + scale) + shift
            if original:
                head = pdd["video_out.weight"][step]
                bias = pdd["video_out.bias"][step]
            else:
                all_heads = weights["final_layer.video_out.weight"].reshape(-1, 96, target.shape[-1])
                all_biases = weights["final_layer.video_out.bias"].reshape(-1, 96)
                head, bias = all_heads[0], all_biases[0]
                if len(all_heads) > 1 and step:
                    head, bias = head + all_heads[step], bias + all_biases[step]
            patches = patches + (sigma - next_sigma) * (target @ head.T + bias)
            mx.eval(patches)
        result = patches.reshape(rows, columns, 24, 2, 2).transpose(2, 0, 3, 1, 4).reshape(24, rows * 2, columns * 2)
        mx.eval(result)
        return result
    finally:
        weights.close()
        if pdd:
            pdd.close()
