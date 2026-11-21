import argparse
import hashlib
import json
import time
import uuid
from pathlib import Path

from onyx import __version__
from onyx.artifacts import artifact_from_path, message_digest, sha256_file, write_proto_json
from onyx.common.v1 import types_pb2
from onyx.model.v1 import model_pb2
from onyx.plan.v1 import plan_pb2
from onyx.replay import source_tree_identity
from onyx.trace.v1 import trace_pb2


ROOT = Path(__file__).resolve().parents[2]


def build_plan(result, metallib: Path):
    shape = result["shape"]
    m, n, k, group = shape["m"], shape["n"], shape["k"], shape["group_size"]
    variant = result.get("variant", "tile16")
    kernel_name = "onyx_q4_gemm_fp16acc32" + ("_tile32" if variant == "tile32" else "")
    tile = 32 if variant == "tile32" else 16
    signature = f"onyx.q4_gemm.v1:{m}:{n}:{k}:{group}".encode()
    implementation = bytes.fromhex(sha256_file(metallib))
    plan = plan_pb2.ExecutionPlan(
        protocol="onyx.plan.v1",
        semantic_digest=hashlib.sha256(signature).digest(),
        target_id="mac17-14-m5-max-36gb",
        backend="metal4",
        numeric_profile_id="fp16.activation.fp32.accumulation",
        quantization_profile_id=f"onyx.q4s{group}.v1",
        kernels=[plan_pb2.KernelVariant(
            variant_id=kernel_name, op="q4_gemm", op_version=1,
            entry_point=kernel_name, implementation_sha256=implementation,
            parameters={"tile_m": str(tile), "tile_n": str(tile), "tile_k": "16"},
            required_features=["metal4", "shared-memory"],
        )],
    )
    buffer_specs = [
        ("activation", m * k * 2, types_pb2.DTYPE_FP16, types_pb2.DTYPE_FP16, [k, 1]),
        ("weights", n * k // 2, types_pb2.DTYPE_FP32, types_pb2.DTYPE_UINT8, [k, 1]),
        ("scales", n * (k // group) * 4, types_pb2.DTYPE_FP32, types_pb2.DTYPE_FP32, [k // group, 1]),
        ("output", m * n * 2, types_pb2.DTYPE_FP16, types_pb2.DTYPE_FP16, [n, 1]),
    ]
    for name, byte_size, logical_dtype, storage_dtype, strides in buffer_specs:
        logical = model_pb2.LogicalTensorType(dtype=logical_dtype)
        if name == "activation":
            logical.shape.extend([model_pb2.Dimension(literal=m), model_pb2.Dimension(literal=k)])
        elif name == "weights":
            logical.shape.extend([model_pb2.Dimension(literal=n), model_pb2.Dimension(literal=k)])
        elif name == "scales":
            logical.shape.extend([model_pb2.Dimension(literal=n), model_pb2.Dimension(literal=k // group)])
        else:
            logical.shape.extend([model_pb2.Dimension(literal=m), model_pb2.Dimension(literal=n)])
        tensor = plan_pb2.PhysicalTensorDesc(
            logical_type=logical, storage_dtype=storage_dtype, strides=strides,
            alignment=64, memory_space=plan_pb2.MEMORY_SPACE_SHARED,
        )
        if name == "weights":
            tensor.quantization.bits = 4
            tensor.quantization.group_size = group
            tensor.quantization.signed_values = True
            tensor.quantization.scale_dtype = types_pb2.DTYPE_FP32
            tensor.quantization.packing = "unsigned-nibble(q+8); low=even-k; high=odd-k"
            tensor.quantization.scales_buffer = "scales"
        plan.tensors.append(tensor)
        plan.buffers.append(plan_pb2.BufferResource(
            buffer_id=name, byte_size=byte_size, memory_space=plan_pb2.MEMORY_SPACE_SHARED,
            alignment=64, tensor=tensor,
        ))
    plan.phases.append(plan_pb2.ResourcePhase(
        phase_id="q4_gemm", resident_buffers=["activation", "weights", "scales", "output"],
        temporary_arena_bytes=0,
    ))
    plan.nodes.append(plan_pb2.PlannedNode(
        node_id="q4_gemm", source_node_ids=["q4_gemm"], kernel_variant_id=kernel_name,
        input_buffers=["activation", "weights", "scales"], output_buffers=["output"], phase_id="q4_gemm",
    ))
    plan.cache_key = message_digest(plan)
    plan.plan_id = f"sha256:{plan.cache_key.hex()}"
    return plan


def record_metal(result_path: Path, output: Path, metallib: Path):
    result = json.loads(result_path.read_text())
    if result.get("operator") != "onyx.q4_gemm.v1":
        raise ValueError("Unsupported Metal benchmark result")
    revision, dirty, tree_digest = source_tree_identity()
    target = json.loads((ROOT / "config/targets/m5-max-36gb.json").read_text())
    shape = result["shape"]
    plan = build_plan(result, metallib)
    output.mkdir(parents=True, exist_ok=True)
    (output / "plan.pb").write_bytes(plan.SerializeToString(deterministic=True))
    write_proto_json(plan, output / "plan.json")
    run_id = str(uuid.uuid4())
    record = trace_pb2.RunRecord(
        protocol="onyx.trace.v1",
        run_id=run_id,
        request_id=f"onyx.q4_gemm.v1:m{shape['m']}:n{shape['n']}:k{shape['k']}",
        plan_id=plan.plan_id,
        numeric_profile_id="fp16.activation.fp32.accumulation",
        quantization_profile_id=f"onyx.q4s{shape['group_size']}.v1",
        build=trace_pb2.BuildIdentity(
            source_revision=revision,
            source_dirty=dirty,
            runtime_version=__version__,
            backend_version="metal4",
            compiler_version=target["clang_version"],
            metallib_sha256=bytes.fromhex(sha256_file(metallib)),
            source_tree_sha256=tree_digest,
            kernel_variant_ids=[plan.kernels[0].variant_id],
        ),
        device=types_pb2.DeviceCapabilities(
            target_id=target["target_id"], model_identifier=target["model_identifier"], chip=target["chip"],
            cpu_cores=target["cpu_cores"], gpu_cores=target["gpu_cores"],
            physical_memory_bytes=target["physical_memory_bytes"],
            recommended_working_set_bytes=result["recommended_working_set_bytes"],
            max_buffer_bytes=result["max_buffer_bytes"], unified_memory=True,
            metal_version=target["metal"], os_version=target["macos_version"], os_build=target["macos_build"],
        ),
        cache_state=trace_pb2.CACHE_STATE_WARM,
        total_wall_time_ns=round(result["wall_median_ms"] * 1e6),
        peak_memory_bytes=result["metal_allocated_bytes"],
        command=result.get("arguments", []),
    )
    stage = record.stages.add()
    stage.stage.stage_id = "q4_gemm"
    stage.gpu_time_ns = round(result["gpu_median_ms"] * 1e6)
    stage.wall_time_ns = round(result["wall_median_ms"] * 1e6)
    stage.peak_memory_bytes = result["metal_allocated_bytes"]
    for name in ("gpu_median_ms", "gpu_p95_ms", "wall_median_ms", "wall_p95_ms", "effective_tflops", "effective_bandwidth_gb_per_s", "max_abs_error", "max_relative_error"):
        unit = "ms" if name.endswith("_ms") else "ratio" if "error" in name else "count/s"
        record.metrics.add(name=name, value=result[name], unit=unit, source="metal-command-buffer")
    for index, value in enumerate(result.get("gpu_samples_ms", [])):
        record.metrics.add(name="gpu_sample", value=value, unit="ms", source=f"iteration:{index}")
    record.outputs.append(artifact_from_path(result_path, types_pb2.ARTIFACT_KIND_TRACE, "application/json", result_path))
    record.outputs.append(artifact_from_path(output / "plan.pb", types_pb2.ARTIFACT_KIND_PLAN, "application/x-protobuf", output / "plan.pb"))
    accepted = record.events.add()
    accepted.run_id = run_id
    accepted.request_id = record.request_id
    accepted.sequence = 0
    accepted.monotonic_time_ns = time.monotonic_ns()
    accepted.wall_time_unix_ns = time.time_ns()
    accepted.accepted.job_id = run_id
    terminal = record.events.add()
    terminal.run_id = run_id
    terminal.request_id = record.request_id
    terminal.sequence = 1
    terminal.monotonic_time_ns = time.monotonic_ns()
    terminal.wall_time_unix_ns = time.time_ns()
    if result["correct"]:
        terminal.completed.SetInParent()
    else:
        terminal.failed.error.code = types_pb2.ERROR_CODE_NUMERIC_FAILURE
        terminal.failed.error.retryability = types_pb2.RETRYABILITY_NO
        terminal.failed.error.message = "Q4 GEMM numerical comparison failed"
        record.terminal_error.CopyFrom(terminal.failed.error)
    output.mkdir(parents=True, exist_ok=True)
    (output / "run.pb").write_bytes(record.SerializeToString(deterministic=True))
    write_proto_json(record, output / "run.json")
    print(json.dumps({"run_id": run_id, "correct": result["correct"], "record": str(output / "run.pb")}, indent=2))
    if not result["correct"]:
        raise RuntimeError("Cannot record a numerically invalid prototype as passing")
    return record


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="onyx metal-record")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metallib", type=Path, default=Path(".build/onyx_q4.metallib"))
    args = parser.parse_args(argv)
    record_metal(args.input, args.output, args.metallib)


if __name__ == "__main__":
    main()
