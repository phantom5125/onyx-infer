import json

import numpy as np
from safetensors.numpy import save_file

from onyx.quantize import PackReader, quantize_component


def make_component(path):
    component = path / "transformer"
    component.mkdir(parents=True)
    rng = np.random.default_rng(42)
    linear = rng.standard_normal((8, 128), dtype=np.float32)
    linear[0] = 0
    norm = np.ones(128, dtype=np.float32)
    save_file({"blocks.0.attn.qkv_proj.weight": linear, "blocks.0.norm1.weight": norm}, str(component / "model.safetensors"))
    index = {
        "metadata": {"total_size": linear.nbytes + norm.nbytes},
        "weight_map": {
            "blocks.0.attn.qkv_proj.weight": "model.safetensors",
            "blocks.0.norm1.weight": "model.safetensors",
        },
    }
    (component / "model.safetensors.index.json").write_text(json.dumps(index))
    return linear, norm


def test_q4_pack_round_trip_and_raw_passthrough(tmp_path):
    source = tmp_path / "source"
    linear, norm = make_component(source)
    output = tmp_path / "transformer.onxpack"
    report = quantize_component(source, "transformer", output, [], all_linear=True, group_size=64)
    pack = PackReader(output)
    entry, packed, scales = pack.q4_arrays("blocks.0.attn.qkv_proj.weight")
    low = (packed & 15).astype(np.int8) - 8
    high = (packed >> 4).astype(np.int8) - 8
    quantized = np.stack((low, high), axis=2).reshape(8, 128)
    restored = quantized.reshape(8, 2, 64).astype(np.float32) * np.repeat(scales, 64, axis=1).reshape(8, 2, 64)
    restored = restored.reshape(8, 128)
    assert np.array_equal(restored[0], np.zeros(128))
    assert np.max(np.abs(restored - linear)) <= np.max(scales) / 2 + 1e-6
    assert report["complete_component"] is True
    assert report["tensor_count"] == 2
    raw = pack.tensors["blocks.0.norm1.weight"]
    restored_norm = pack.array(raw, "data_offsets", np.float32)
    assert np.array_equal(np.asarray(restored_norm), norm)
    assert entry["source_sha256"] == report["tensors"][0]["source_sha256"]
