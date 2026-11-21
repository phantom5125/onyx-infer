import json

import numpy as np
from safetensors.numpy import save_file

from onyx.importer import import_h3
from onyx.model.v1 import model_pb2
from onyx.quantize import quantize_component
from onyx.w4_package import build_package


ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def make_weight(source, component, name):
    directory = source / component
    directory.mkdir(parents=True)
    weight = np.random.default_rng(42).standard_normal((8, 128), dtype=np.float32)
    save_file({name: weight}, str(directory / "model.safetensors"))
    (directory / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {name: "model.safetensors"}}))


def test_w4_package_has_distinct_provenance(tmp_path, h3_test_config):
    source = tmp_path / "source"
    output = tmp_path / "w4"
    make_weight(source, "transformer", "blocks.0.attn.qkv_proj.weight")
    make_weight(source, "text_encoder", "model.layers.0.self_attn.q_proj.weight")
    quantize_component(source, "transformer", output / "transformer.onxpack", [], all_linear=True)
    quantize_component(source, "text_encoder", output / "text_encoder.onxpack", [], all_linear=True)
    original, _ = import_h3(h3_test_config, ROOT / "config/sources/h3-image-files.json", tmp_path / "original", metadata_only=True)
    manifest = build_package(source, tmp_path / "original/manifest.pb", output)
    decoded = model_pb2.ModelManifest.FromString((output / "manifest.pb").read_bytes())
    assert decoded == manifest
    assert decoded.source_model_semantic_digest == original.semantic_digest
    assert decoded.semantic_digest != original.semantic_digest
    assert decoded.quantization.profile_id == "onyx-w4s64-linear-v1"
    assert len(decoded.quantization.tensor_rules) == 11
    assert decoded.components[0].artifacts[0].byte_size > 0
    assert (output / "pdd_config.json").is_symlink()
