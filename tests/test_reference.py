import numpy as np
from safetensors.numpy import save_file

mx = __import__("pytest").importorskip("mlx.core")


def test_convrot_int8_linear_matches_unquantized_projection(tmp_path):
    from onyx.reference.weights import Weights, rotation

    rng = np.random.default_rng(42)
    original = rng.standard_normal((16, 256), dtype=np.float32) * 0.2
    h = np.asarray(rotation())
    rotated = (original.reshape(16, 1, 256) @ h).reshape(16, 256)
    scale = np.max(np.abs(rotated), axis=1, keepdims=True) / 127
    quantized = np.clip(np.rint(rotated / scale), -127, 127).astype(np.int8)
    save_file({"projection.weight": quantized, "projection.weight_scale": scale.astype(np.float32)}, str(tmp_path / "weights.safetensors"))
    activation = mx.array(rng.standard_normal((4, 256), dtype=np.float32).astype(np.float16))
    weights = Weights(tmp_path / "weights.safetensors")
    actual = np.asarray(weights.linear("projection", activation).astype(mx.float32))
    weights.close()
    expected = np.asarray(activation).astype(np.float32) @ original.T
    assert np.max(np.abs(actual - expected)) < 0.08


def test_vae_tile_grid_preserves_boundaries_and_overlap():
    from onyx.reference.vae import tile_positions

    assert tile_positions(16) == [0]
    assert tile_positions(64) == [0, 12, 24, 36, 48]
    assert all(position + 16 <= 64 for position in tile_positions(64))


def test_onxpack_q4_linear_uses_the_same_quantized_weights(tmp_path):
    import json

    from onyx.quantize import PackReader, quantize_component
    from onyx.reference.weights import Weights

    rng = np.random.default_rng(7)
    source = tmp_path / "source"
    component = source / "transformer"
    component.mkdir(parents=True)
    original = rng.standard_normal((8, 128), dtype=np.float32)
    name = "blocks.0.attn.qkv_proj.weight"
    save_file({name: original}, str(component / "model.safetensors"))
    (component / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {name: "model.safetensors"}}))
    pack_path = tmp_path / "weights.onxpack"
    quantize_component(source, "transformer", pack_path, [], all_linear=True)
    activation_np = rng.standard_normal((4, 128), dtype=np.float32)
    activation = mx.array(activation_np.astype(np.float16))
    weights = Weights(pack_path, resident=True)
    actual = np.asarray(weights.linear("blocks.0.attn.qkv_proj", activation).astype(mx.float32))
    weights.close()
    pack = PackReader(pack_path)
    _, packed, scales = pack.q4_arrays(name)
    low = (packed & 15).astype(np.int8) - 8
    high = (packed >> 4).astype(np.int8) - 8
    quantized = np.stack((low, high), axis=2).reshape(8, 128)
    restored = quantized.reshape(8, 2, 64).astype(np.float32) * np.repeat(scales, 64, axis=1).reshape(8, 2, 64)
    expected = activation_np.astype(np.float16).astype(np.float32) @ restored.reshape(8, 128).T
    assert np.max(np.abs(actual - expected)) < 0.03
    raw_weights = Weights(pack_path, raw_source=source)
    raw_actual = np.asarray(raw_weights.linear("blocks.0.attn.qkv_proj", activation).astype(mx.float32))
    raw_weights.close()
    raw_expected = activation_np.astype(np.float16).astype(np.float32) @ original.astype(np.float16).astype(np.float32).T
    assert np.max(np.abs(raw_actual - raw_expected)) < 0.01


def test_original_timestep_embedding_boundaries():
    from onyx.reference.dit import timestep_projection

    values = np.asarray(timestep_projection([0.0, 1.0]))
    assert values.shape == (2, 256)
    assert values.dtype == np.float32
    np.testing.assert_allclose(values[0, :128], 1.0, atol=1e-7)
    np.testing.assert_allclose(values[0, 128:], 0.0, atol=1e-7)
    assert np.isfinite(values).all()
    assert not np.array_equal(values[0], values[1])


def test_original_qkv_layout_reorders_head_interleaved_rows():
    from onyx.reference.dit import reorder_original_qkv

    value = np.arange(56 * 3 * 128, dtype=np.float32).reshape(1, 56, 3, 128)
    expected = value.transpose(0, 2, 1, 3).reshape(1, 56 * 3 * 128)
    actual = reorder_original_qkv(value.reshape(1, 56 * 3 * 128))
    np.testing.assert_array_equal(np.asarray(actual), expected)
