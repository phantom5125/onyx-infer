from pathlib import Path

import mlx.core as mx
from tokenizers import Tokenizer

from onyx.reference.weights import Weights, attention, heads, norm


def encode(weights_path, tokenizer_path, prompt, original=False):
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    ids = tokenizer.encode(prompt, add_special_tokens=False).ids[:4096]
    if not ids:
        raise ValueError("Empty prompt after tokenization")
    weights = Weights(weights_path)
    try:
        model_prefix = "model.language_model." if original else "model."
        x = weights.embedding(model_prefix + "embed_tokens.weight", ids)
        prefixes = [key for key in weights.header if key.startswith(model_prefix + "layers.") and key.endswith(".input_layernorm.weight")]
        layer_count = 50 if original else len(prefixes)
        for layer in range(layer_count):
            prefix = f"{model_prefix}layers.{layer}."
            h = norm(x, weights[prefix + "input_layernorm.weight"], 1e-6)
            attention_prefix = prefix + "self_attn."
            q, k, v = (heads(weights.linear(attention_prefix + name + "_proj", h)) for name in ("q", "k", "v"))
            q = norm(q, weights[attention_prefix + "q_norm.weight"], 1e-6)
            k = norm(k, weights[attention_prefix + "k_norm.weight"], 1e-6)
            q = mx.fast.rope(q, 128, traditional=False, base=5_000_000, scale=1.0, offset=0)
            k = mx.fast.rope(k, 128, traditional=False, base=5_000_000, scale=1.0, offset=0)
            x = x + weights.linear(attention_prefix + "o_proj", attention(q, k, v, mask="causal"))
            h = norm(x, weights[prefix + "post_attention_layernorm.weight"], 1e-6)
            gate = weights.linear(prefix + "mlp.gate_proj", h)
            up = weights.linear(prefix + "mlp.up_proj", h)
            x = x + weights.linear(prefix + "mlp.down_proj", gate * mx.sigmoid(gate) * up)
            mx.eval(x)
        return x, ids
    finally:
        weights.close()
