import json
import subprocess
import sys
from pathlib import Path

import pytest
from google.protobuf import json_format

from onyx.common.v1 import types_pb2
from onyx.job.v1 import job_pb2
from onyx.loop.v1 import loop_pb2


ROOT = Path(__file__).resolve().parents[1]


def test_benchmark_request_parses_and_round_trips():
    source = json.loads((ROOT / "benchmarks/requests/teapot-1024.json").read_text())
    case = loop_pb2.BenchmarkCase()
    json_format.ParseDict(source, case, ignore_unknown_fields=False)
    assert case.protocol == "onyx.loop.v1"
    assert case.request.protocol == "onyx.job.v1"
    assert case.request.seed == 42
    assert case.request.size.width == case.request.size.height == 1024
    assert case.request.output.format == job_pb2.OutputSpec.PNG
    encoded = case.SerializeToString(deterministic=True)
    decoded = loop_pb2.BenchmarkCase.FromString(encoded)
    assert decoded == case
    assert json_format.MessageToDict(decoded)["request"]["seed"] == "42"


def test_unknown_request_field_is_rejected():
    request = job_pb2.GenerateRequest()
    with pytest.raises(json_format.ParseError):
        json_format.ParseDict({"guidance_scale": 1.0}, request, ignore_unknown_fields=False)


def test_event_has_one_terminal_payload():
    event = job_pb2.RunEvent(sequence=0)
    event.completed.run_record.artifact_id = "sha256:test"
    assert event.WhichOneof("payload") == "completed"
    event.failed.error.code = types_pb2.ERROR_CODE_INTERNAL
    assert event.WhichOneof("payload") == "failed"


def test_device_contract_preserves_memory_budget():
    target = json.loads((ROOT / "config/targets/m5-max-36gb.json").read_text())
    device = types_pb2.DeviceCapabilities(
        target_id=target["target_id"],
        physical_memory_bytes=target["physical_memory_bytes"],
        recommended_working_set_bytes=target["recommended_gpu_working_set_bytes"],
        max_buffer_bytes=target["max_metal_buffer_bytes"],
        unified_memory=target["unified_memory"],
    )
    assert device.recommended_working_set_bytes == 30150672384
    assert device.recommended_working_set_bytes < device.physical_memory_bytes


def test_schema_compatibility_baseline():
    subprocess.run([sys.executable, str(ROOT / "tools/check_schema.py")], check=True)
