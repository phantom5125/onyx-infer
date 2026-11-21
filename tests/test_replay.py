import json
import subprocess
import sys
from pathlib import Path

import psutil

from onyx.replay import ProcessMonitor
from onyx.trace.v1 import trace_pb2


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = """import hashlib, os
from PIL import Image
payload = (os.environ['ONYX_PROMPT'] + os.environ['ONYX_SEED']).encode()
color = tuple(hashlib.sha256(payload).digest()[:3])
Image.new('RGB', (int(os.environ['ONYX_WIDTH']), int(os.environ['ONYX_HEIGHT'])), color).save(os.environ['ONYX_OUTPUT'])
"""


def run_replay(output_root, expected=None, target=None):
    command = [
        sys.executable, "-m", "onyx.replay",
        "--case", str(ROOT / "benchmarks/requests/teapot-1024.json"),
        "--output-root", str(output_root),
        "--backend-version", "test-generator-v1",
        "--numeric-profile", "test-rgb",
        "--rng-algorithm", "numpy.pcg64.standard_normal",
        "--rng-version", "test",
        "--cache-state", "cold",
    ]
    if expected:
        command += ["--expect-output-sha256", expected]
    if target is not None:
        command += ["--target", str(target)]
    command += ["--", sys.executable, "-c", GENERATOR]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_real_process_replay_and_output_verification(tmp_path):
    first = run_replay(tmp_path)
    second = run_replay(tmp_path, first["output_sha256"])
    assert first["return_code"] == second["return_code"] == 0
    assert first["output_sha256"] == second["output_sha256"]
    record = trace_pb2.RunRecord.FromString(Path(second["run_record"]).read_bytes())
    assert record.protocol == "onyx.trace.v1"
    assert record.replay_request.seed == 42
    assert record.rng.actual_seed == 42
    assert len(record.request_digest) == 32
    assert len(record.build.source_tree_sha256) == 32
    assert record.events[-1].WhichOneof("payload") == "completed"
    assert record.outputs[0].sha256.hex() == first["output_sha256"]
    assert record.total_wall_time_ns > 0


def test_process_monitor_samples_live_process():
    with ProcessMonitor(psutil.Process()) as monitor:
        assert monitor.samples >= 1
        assert monitor.peak_rss > 0


def test_process_monitor_keeps_rss_when_vm_query_fails(monkeypatch):
    process = psutil.Process()

    def unavailable():
        raise RuntimeError("mach_vm_region(VM_REGION_TOP_INFO) syscall failed")

    monkeypatch.setattr(process, "memory_full_info", unavailable)
    rss, footprint = ProcessMonitor.memory_bytes(process)
    assert rss > 0
    assert footprint == 0


def test_replay_records_selected_device_instead_of_reference_machine(tmp_path):
    target = json.loads((ROOT / "config/targets/m5-max-36gb.json").read_text())
    target.update(target_id="local-m4", model_identifier="Mac16,12", chip="Apple M4",
                  cpu_cores=10, gpu_cores=10, physical_memory_bytes=24 * 1024**3)
    target_path = tmp_path / "target.json"
    target_path.write_text(json.dumps(target))
    run = run_replay(tmp_path / "runs", target=target_path)
    record = trace_pb2.RunRecord.FromString(Path(run["run_record"]).read_bytes())
    assert record.device.target_id == "local-m4"
    assert record.device.chip == "Apple M4"
    assert record.device.physical_memory_bytes == 24 * 1024**3
    assert record.device.gpu_cores == 10
