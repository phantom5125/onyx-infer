import json
from pathlib import Path

from onyx.artifacts import artifact_from_lock, message_digest
from onyx.common.v1 import types_pb2
from onyx.model.v1 import model_pb2


def attribute(value):
    result = model_pb2.Attribute()
    if isinstance(value, bool):
        result.boolean = value
    elif isinstance(value, int):
        result.integer = value
    elif isinstance(value, float):
        result.floating = value
    elif isinstance(value, str):
        result.text = value
    elif isinstance(value, list) and all(isinstance(item, int) and not isinstance(item, bool) for item in value):
        result.integers.values.extend(value)
    elif isinstance(value, list) and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value):
        result.floating_values.values.extend(value)
    elif isinstance(value, list) and all(isinstance(item, str) for item in value):
        result.strings.values.extend(value)
    else:
        result.text = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return result


def attributes(config):
    return {key: attribute(value) for key, value in config.items() if not key.startswith("_")}


def dimension(value):
    return model_pb2.Dimension(literal=value) if isinstance(value, int) else model_pb2.Dimension(symbol=value)


def tensor_type(dtype, shape):
    return model_pb2.LogicalTensorType(dtype=dtype, shape=[dimension(item) for item in shape])


def value(name, dtype, shape):
    return model_pb2.Value(value_id=name, type=tensor_type(dtype, shape))


def node(node_id, op, inputs, outputs, config=None, **extra):
    attrs = attributes(config or {})
    attrs.update({key: attribute(item) for key, item in extra.items()})
    return model_pb2.Node(node_id=node_id, op=op, op_version=1, inputs=inputs, outputs=outputs, attributes=attrs)


def build_h3_manifest(config_root: Path, lock):
    dit_config = json.loads((config_root / "transformer/config.json").read_text())
    text_config = json.loads((config_root / "text_encoder/config.json").read_text())
    vae_config = json.loads((config_root / "vae/config.json").read_text())
    pdd_config = json.loads((config_root / "pdd_config.json").read_text())

    text_graph = model_pb2.SemanticGraph(
        graph_id="h3.qwen3vl.text50",
        values=[
            value("input_ids", types_pb2.DTYPE_INT64, [1, "text_tokens"]),
            value("conditioning", types_pb2.DTYPE_BF16, ["text_tokens", 5120]),
        ],
        nodes=[node("qwen3vl_text50", "qwen3_vl_text", ["input_ids"], ["conditioning"], text_config, output_layer=50)],
        inputs=["input_ids"],
        outputs=["conditioning"],
        symbolic_constraints_min={"text_tokens": 1},
        symbolic_constraints_max={"text_tokens": 4096},
    )

    dit_values = [
        value("conditioning", types_pb2.DTYPE_BF16, ["text_tokens", 5120]),
        value("latent", types_pb2.DTYPE_FP32, [24, "latent_height", "latent_width"]),
        value("sigma", types_pb2.DTYPE_FP32, []),
        value("refined", types_pb2.DTYPE_BF16, ["text_tokens", 5376]),
        value("state_0", types_pb2.DTYPE_BF16, ["packed_tokens", 5376]),
        value("denoised_latent", types_pb2.DTYPE_FP32, [24, "latent_height", "latent_width"]),
    ]
    dit_values.extend(value(f"state_{index}", types_pb2.DTYPE_BF16, ["packed_tokens", 5376]) for index in range(1, 51))
    dit_nodes = [
        node("prepare_condition", "h3_prepare_condition", ["conditioning"], ["refined"], dit_config),
        node("pack_latent", "h3_pack_latent", ["latent", "refined"], ["state_0"], dit_config),
    ]
    for index in range(50):
        dit_nodes.append(node(f"dit_block_{index}", "h3_dit_block", [f"state_{index}", "sigma"], [f"state_{index + 1}"], dit_config, layer_index=index))
    dit_nodes.append(node("pdd_output", "h3_pdd_output", ["state_50", "sigma"], ["denoised_latent"], dit_config))
    dit_graph = model_pb2.SemanticGraph(
        graph_id="h3.dit.step",
        values=dit_values,
        nodes=dit_nodes,
        inputs=["conditioning", "latent", "sigma"],
        outputs=["denoised_latent"],
        symbolic_constraints_min={"latent_height": 16, "latent_width": 16, "packed_tokens": 1},
    )

    vae_graph = model_pb2.SemanticGraph(
        graph_id="h3.vae.decode",
        values=[
            value("latent", types_pb2.DTYPE_FP32, [24, "latent_height", "latent_width"]),
            value("image", types_pb2.DTYPE_FP32, [3, "image_height", "image_width"]),
        ],
        nodes=[node("vae_decode", "h3_vae_decode_2d", ["latent"], ["image"], vae_config)],
        inputs=["latent"],
        outputs=["image"],
    )

    components = {
        "text_encoder": model_pb2.Component(component_id="text_encoder", role=model_pb2.COMPONENT_ROLE_TEXT_ENCODER, graph_id=text_graph.graph_id),
        "transformer": model_pb2.Component(component_id="transformer", role=model_pb2.COMPONENT_ROLE_TRANSFORMER, graph_id=dit_graph.graph_id),
        "vae": model_pb2.Component(component_id="vae", role=model_pb2.COMPONENT_ROLE_VAE, graph_id=vae_graph.graph_id),
        "tokenizer": model_pb2.Component(component_id="tokenizer", role=model_pb2.COMPONENT_ROLE_TOKENIZER),
        "pdd": model_pb2.Component(component_id="pdd", role=model_pb2.COMPONENT_ROLE_AUXILIARY),
    }
    for item in lock["files"]:
        path = item["path"]
        if path.startswith("text_encoder/"):
            component = components["text_encoder"]
        elif path.startswith("transformer/"):
            component = components["transformer"]
        elif path.startswith("vae/"):
            component = components["vae"]
        elif path.startswith(("tokenizer/", "processor/")):
            component = components["tokenizer"]
        elif path.startswith("pdd_"):
            component = components["pdd"]
        else:
            continue
        component.artifacts.append(artifact_from_lock(item))

    recipe = model_pb2.Recipe(
        recipe_id="h3.pdd6.v1",
        calls=[
            model_pb2.StageCall(stage_id="condition", iterations=1),
            model_pb2.StageCall(stage_id="sample", iterations=6),
            model_pb2.StageCall(stage_id="decode", iterations=1),
        ],
        sigmas=pdd_config["sigmas"],
        rng=model_pb2.RngSpec(
            algorithm="numpy.pcg64.standard_normal",
            algorithm_version="numpy>=2.2,<3",
            generator_device="cpu",
            noise_type=tensor_type(types_pb2.DTYPE_FP32, [24, "latent_height", "latent_width"]),
            noise_layout="CHW row-major; NumPy Generator PCG64",
        ),
        numeric_profile_id="h3-w4a16-fp32accum",
        request_constraints={
            "width_multiple": attribute(32),
            "height_multiple": attribute(32),
            "num_inference_steps": attribute(6),
            "guidance_scale": attribute(1.0),
            "reference_images_max": attribute(16),
        },
    )
    numeric = model_pb2.NumericProfile(
        profile_id="h3-w4a16-fp32accum",
        activation_dtype=types_pb2.DTYPE_BF16,
        accumulation_dtype=types_pb2.DTYPE_FP32,
        softmax_dtype=types_pb2.DTYPE_FP32,
        normalization_dtype=types_pb2.DTYPE_FP32,
        fast_math=False,
    )
    manifest = model_pb2.ModelManifest(
        protocol="onyx.model.v1",
        model_id="minimax-h3-image",
        source_revision=lock["revision"],
        components=list(components.values()),
        graphs=[text_graph, dit_graph, vae_graph],
        stages=[
            model_pb2.Stage(stage_id="condition", component_id="text_encoder", graph_id=text_graph.graph_id, inputs=["input_ids"], outputs=["conditioning"]),
            model_pb2.Stage(stage_id="sample", component_id="transformer", graph_id=dit_graph.graph_id, inputs=["conditioning", "latent", "sigma"], outputs=["denoised_latent"]),
            model_pb2.Stage(stage_id="decode", component_id="vae", graph_id=vae_graph.graph_id, inputs=["latent"], outputs=["image"]),
        ],
        recipes=[recipe],
        numeric_profile_id=numeric.profile_id,
        numeric_profiles=[numeric],
    )
    manifest.semantic_digest = message_digest(manifest)
    return manifest
