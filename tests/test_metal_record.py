import json

from onyx.metal import record_metal
from onyx.plan.v1 import plan_pb2
from onyx.trace.v1 import trace_pb2


def test_metal_result_produces_content_addressed_plan(tmp_path):
    metallib = tmp_path / "ops.metallib"
    metallib.write_bytes(b"test-metal-library")
    result = {
        "operator": "onyx.q4_gemm.v1",
        "shape": {"m": 16, "n": 128, "k": 256, "group_size": 64},
        "correct": True,
        "recommended_working_set_bytes": 30150672384,
        "max_buffer_bytes": 22613000192,
        "wall_median_ms": 0.2,
        "gpu_median_ms": 0.1,
        "metal_allocated_bytes": 1234,
        "gpu_p95_ms": 0.1,
        "wall_p95_ms": 0.2,
        "effective_tflops": 1.0,
        "effective_bandwidth_gb_per_s": 1.0,
        "max_abs_error": 0.0,
        "max_relative_error": 0.0,
        "gpu_samples_ms": [0.1],
        "arguments": ["onyx-metal-bench"],
    }
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(result))
    record_metal(result_path, tmp_path / "record", metallib)
    plan = plan_pb2.ExecutionPlan.FromString((tmp_path / "record/plan.pb").read_bytes())
    record = trace_pb2.RunRecord.FromString((tmp_path / "record/run.pb").read_bytes())
    assert plan.protocol == "onyx.plan.v1"
    assert plan.plan_id == f"sha256:{plan.cache_key.hex()}"
    assert len(plan.cache_key) == 32
    assert plan.nodes[0].input_buffers == ["activation", "weights", "scales"]
    assert plan.buffers[1].tensor.quantization.group_size == 64
    assert record.plan_id == plan.plan_id
    assert record.events[-1].WhichOneof("payload") == "completed"
