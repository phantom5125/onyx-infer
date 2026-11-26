# Portions adapted from Anionex/h3-image-mlx (eccdf32), licensed under
# Apache-2.0. See LICENSE.h3-image-mlx.
"""H3's Qwen3-VL visual presentation, including DeepStack and 3-axis text RoPE."""

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from PIL import Image
from onyx.reference.weights import attention, heads, rope


def reference_image(image, max_short_edge):
    short = min(image.size)
    scale = min(max(short, 1024), max_short_edge) / short
    size = tuple(round(dimension * scale / 32) * 32 for dimension in image.size)
    return np.asarray(image.resize(size, Image.Resampling.LANCZOS), dtype=np.float32) / 255


def visual_features(w, image):
    im = reference_image(image, 1024) * 2 - 1
    gh, gw = im.shape[0] // 16, im.shape[1] // 16
    index = (
        np.arange(gh * gw)
        .reshape(gh // 2, 2, gw // 2, 2)
        .transpose(0, 2, 1, 3)
        .reshape(-1)
    )
    pixels = (
        im.reshape(gh, 16, gw, 16, 3).transpose(0, 2, 4, 1, 3).reshape(-1, 768)[index]
    )
    # A still image is repeated for both temporal taps of the patch embedding.
    proj = (
        w["model.visual.patch_embed.proj.weight"]
        .astype(mx.float32)
        .sum(axis=2)
        .reshape(1152, 768)
    )
    x = (mx.array(pixels) @ proj.T + w["model.visual.patch_embed.proj.bias"]).astype(
        mx.bfloat16
    )
    yy, xx = np.unravel_index(index, (gh, gw))
    coord = np.stack([yy * 47 / (gh - 1), xx * 47 / (gw - 1)], axis=-1)
    low = np.floor(coord).astype(int)
    frac = coord - low
    table = w["model.visual.pos_embed.weight"]
    pos = 0
    for dy, dx in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        indices = np.minimum(low[:, 0] + dy, 47) * 48 + np.minimum(low[:, 1] + dx, 47)
        coef = (frac[:, 0] if dy else 1 - frac[:, 0]) * (
            frac[:, 1] if dx else 1 - frac[:, 1]
        )
        pos = pos + table[mx.array(indices)] * mx.array(coef.astype("f4"))[:, None]
    x = x + pos.astype(x.dtype)
    angle = np.stack([yy, xx], axis=1)[:, :, None] * (10000 ** (-np.arange(18) / 18))
    angles = mx.array(angle.reshape(-1, 36).astype(np.float32))

    def ln(x, p):
        return mx.fast.layer_norm(x, w[p + ".weight"], w[p + ".bias"], 1e-6)

    def merge(x, p, post_shuffle):
        if post_shuffle:
            x = x.reshape(-1, 4608)
        x = ln(x, p + ".norm").reshape(-1, 4608)
        return w.linear(p + ".linear_fc2", nn.gelu(w.linear(p + ".linear_fc1", x)))

    deep = []
    for i in range(27):
        p = f"model.visual.blocks.{i}"
        q, k, v = (
            heads(t, 72)
            for t in mx.split(
                w.linear(p + ".attn.qkv", ln(x, p + ".norm1")), 3, axis=-1
            )
        )
        x = x + w.linear(
            p + ".attn.proj", attention(rope(q, angles), rope(k, angles), v)
        )
        h = nn.gelu_approx(w.linear(p + ".mlp.linear_fc1", ln(x, p + ".norm2")))
        x = x + w.linear(p + ".mlp.linear_fc2", h)
        if i in (8, 16, 24):
            deep.append(merge(x, f"model.visual.deepstack_merger_list.{len(deep)}", True))
        mx.eval(x, *deep)
    return merge(x, "model.visual.merger", False), deep, (gh // 2, gw // 2)
