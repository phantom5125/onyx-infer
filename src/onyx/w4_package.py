import argparse
import json
import os
from pathlib import Path

from onyx import __version__
from onyx.artifacts import artifact_from_path, message_digest, write_proto_json
from onyx.common.v1 import types_pb2
from onyx.model.v1 import model_pb2
from onyx.quantize import LINEAR_SUFFIXES


def link(source: Path, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        return
    os.symlink(source.resolve(), destination)


def build_package(source: Path, original_manifest_path: Path, output: Path):
    manifest = model_pb2.ModelManifest.FromString(original_manifest_path.read_bytes())
    transformer_path = output / "transformer.onxpack"
    text_path = output / "text_encoder.onxpack"
    transformer = artifact_from_path(transformer_path, types_pb2.ARTIFACT_KIND_WEIGHTS, "application/x-onyx-pack", transformer_path)
    text = artifact_from_path(text_path, types_pb2.ARTIFACT_KIND_WEIGHTS, "application/x-onyx-pack", text_path)
    for component in manifest.components:
        if component.component_id == "transformer":
            component.ClearField("artifacts")
            component.artifacts.append(transformer)
        elif component.component_id == "text_encoder":
            component.ClearField("artifacts")
            component.artifacts.append(text)
    manifest.source_model_semantic_digest = manifest.semantic_digest
    manifest.semantic_digest = b""
    rules = []
    for suffix in LINEAR_SUFFIXES:
        rules.append(model_pb2.TensorQuantization(
            tensor_pattern=suffix, bits=4, signed_values=True,
            granularity=model_pb2.QUANTIZATION_GRANULARITY_GROUP_WISE, group_size=64,
            scale_dtype=types_pb2.DTYPE_FP32, symmetric=True,
            rounding="nearest-even", clipping="[-7,7]", tensor_ndim=2,
        ))
    manifest.quantization.CopyFrom(model_pb2.QuantizationProfile(
        profile_id="onyx-w4s64-linear-v1",
        algorithm="symmetric-groupwise-maxabs",
        algorithm_version="onyx-q4-v1",
        tool_identity=f"onyx-infer:{__version__}",
        tensor_rules=rules,
        high_precision_exceptions=[
            "all non-selected tensors", "embeddings", "normalization", "timestep MLP",
            "AdaLN projections", "PDD heads", "VAE", "vision and reference paths",
        ],
    ))
    manifest.semantic_digest = message_digest(manifest)
    write_proto_json(manifest, output / "manifest.json")
    (output / "manifest.pb").write_bytes(manifest.SerializeToString(deterministic=True))
    for relative in ("pdd_config.json", "pdd_fused_heads.safetensors", "vae/config.json", "vae/diffusion_pytorch_model.safetensors", "tokenizer/tokenizer.json"):
        link(source / relative, output / relative)
    report = {
        "protocol": "onyx.model-package.v1",
        "model_id": manifest.model_id,
        "source_revision": manifest.source_revision,
        "source_semantic_digest": manifest.source_model_semantic_digest.hex(),
        "semantic_digest": manifest.semantic_digest.hex(),
        "quantization_profile": manifest.quantization.profile_id,
        "transformer_pack": {"path": str(transformer_path), "bytes": transformer.byte_size, "sha256": transformer.sha256.hex()},
        "text_encoder_pack": {"path": str(text_path), "bytes": text.byte_size, "sha256": text.sha256.hex()},
        "quality_status": "pending-end-to-end-validation",
    }
    (output / "package-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(prog="onyx package-w4")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--original-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    build_package(args.source, args.original_manifest, args.output)


if __name__ == "__main__":
    main()
