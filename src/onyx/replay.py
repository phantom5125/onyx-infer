import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from contextlib import nullcontext
from pathlib import Path

import psutil
from google.protobuf import json_format

from onyx import __version__
from onyx.artifacts import artifact_from_path, message_digest, sha256_file, write_proto_json
from onyx.common.v1 import types_pb2
from onyx.job.v1 import job_pb2
from onyx.loop.v1 import loop_pb2
from onyx.trace.v1 import trace_pb2


ROOT = Path(__file__).resolve().parents[2]


def source_tree_identity():
    files = subprocess.check_output(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=ROOT).split(b"\0")
    digest = hashlib.sha256()
    for raw in sorted(item for item in files if item):
        path = Path(os.fsdecode(raw))
        data = (ROOT / path).read_bytes()
        digest.update(len(raw).to_bytes(4, "little"))
        digest.update(raw)
        digest.update(len(data).to_bytes(8, "little"))
        digest.update(hashlib.sha256(data).digest())
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    return revision, dirty, digest.digest()


def load_case(path: Path):
    case = loop_pb2.BenchmarkCase()
    json_format.ParseDict(json.loads(path.read_text()), case, ignore_unknown_fields=False)
    if case.protocol != "onyx.loop.v1" or case.request.protocol != "onyx.job.v1":
        raise ValueError("Unsupported protocol")
    if not case.request.request_id or not case.request.prompt or not case.request.HasField("seed"):
        raise ValueError("Replay requires request_id, prompt and seed")
    if case.request.image_count != 1:
        raise ValueError("The initial replay runner supports one output image")
    if len(case.request.references) > 16:
        raise ValueError("H3 accepts up to 16 reference images")
    if case.request.size.width % 32 or case.request.size.height % 32:
        raise ValueError("H3 image dimensions must be multiples of 32")
    return case


def swap_bytes():
    output = subprocess.check_output(["sysctl", "vm.swapusage"], text=True)
    used = output.split("used = ", 1)[1].split("M", 1)[0]
    return int(float(used) * 1024 * 1024)


class ProcessMonitor:
    def __init__(self, process):
        self.process = process
        self.peak_rss = 0
        self.peak_footprint = 0
        self.samples = 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    @staticmethod
    def memory_bytes(process):
        try:
            rss = process.memory_info().rss
        except psutil.Error:
            return 0, 0
        try:
            footprint = process.memory_full_info().phys_footprint
        except (psutil.Error, AttributeError, OSError, RuntimeError):
            # macOS can reject VM region queries for a child process.
            footprint = 0
        return rss, footprint

    def sample(self):
        rss, footprint = self.memory_bytes(self.process)
        try:
            children = self.process.children(recursive=True)
        except psutil.Error:
            children = ()
        for process in children:
            child_rss, child_footprint = self.memory_bytes(process)
            rss += child_rss
            footprint += child_footprint
        self.peak_rss = max(self.peak_rss, rss)
        self.peak_footprint = max(self.peak_footprint, footprint)
        if rss or footprint:
            self.samples += 1

    def run(self):
        while not self.stop.wait(0.02):
            self.sample()
        self.sample()

    def __enter__(self):
        self.sample()
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()


def execute_case(case: Path, output_root: Path, command, args):
    benchmark = load_case(case)
    request = benchmark.request
    reference_arguments = []
    if request.references:
        if command.count("{references}") != 1:
            raise ValueError("Replay with reference images requires one {references} command argument")
        for ref in request.references:
            if not ref.local_locator or not ref.sha256:
                raise ValueError("Replay requires a local path and SHA-256 for every reference")
            path = (case.parent / ref.local_locator).resolve()
            if path.stat().st_size != ref.byte_size or sha256_file(path) != ref.sha256.hex():
                raise ValueError(f"Reference artifact integrity check failed: {path}")
            reference_arguments.extend(("--reference", str(path)))
    run_id = str(uuid.uuid4())
    run_dir = output_root / benchmark.case_id / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    output = run_dir / args.output_name
    replacements = {
        "output": str(output),
        "run_dir": str(run_dir),
        "prompt": request.prompt,
        "seed": str(request.seed),
        "width": str(request.size.width),
        "height": str(request.size.height),
    }
    actual_command = []
    for part in command:
        if part == "{references}":
            actual_command.extend(reference_arguments)
            continue
        for key, value in replacements.items():
            part = part.replace("{" + key + "}", value)
        actual_command.append(part)
    environment = dict(os.environ)
    environment.update({f"ONYX_{key.upper()}": value for key, value in replacements.items()})
    revision, dirty, tree_digest = source_tree_identity()
    target = json.loads(args.target.read_text())
    device = types_pb2.DeviceCapabilities(
        target_id=target["target_id"], model_identifier=target["model_identifier"], chip=target["chip"],
        cpu_cores=target["cpu_cores"], gpu_cores=target["gpu_cores"],
        physical_memory_bytes=target["physical_memory_bytes"],
        recommended_working_set_bytes=target["recommended_gpu_working_set_bytes"],
        max_buffer_bytes=target["max_metal_buffer_bytes"], unified_memory=target["unified_memory"],
        metal_version=target["metal"], os_version=target["macos_version"], os_build=target["macos_build"],
    )
    record = trace_pb2.RunRecord(
        protocol="onyx.trace.v1", run_id=run_id, request_id=request.request_id,
        request_digest=message_digest(request), resolved_model=request.model,
        plan_id=args.plan_id, numeric_profile_id=args.numeric_profile,
        quantization_profile_id=args.quantization_profile, device=device,
        build=trace_pb2.BuildIdentity(
            source_revision=revision, source_dirty=dirty, runtime_version=__version__,
            backend_version=args.backend_version, source_tree_sha256=tree_digest,
            kernel_variant_ids=args.kernel_variant,
        ),
        rng=trace_pb2.RngExecution(
            algorithm=args.rng_algorithm, algorithm_version=args.rng_version,
            generator_device="cpu", actual_seed=request.seed,
        ),
        cache_state=args.cache_state, command=actual_command,
    )
    if args.resolved_source_revision:
        record.resolved_model.source_revision = args.resolved_source_revision
    if args.manifest_digest:
        record.resolved_model.manifest_digest = bytes.fromhex(args.manifest_digest)
    record.replay_request.CopyFrom(request)
    accepted = record.events.add()
    accepted.run_id = run_id
    accepted.request_id = request.request_id
    accepted.sequence = 0
    accepted.monotonic_time_ns = time.monotonic_ns()
    accepted.wall_time_unix_ns = time.time_ns()
    accepted.accepted.job_id = run_id
    record.wall_start_unix_ns = time.time_ns()
    record.monotonic_start_ns = time.monotonic_ns()
    swap_before = swap_bytes()
    return_code = 0
    error_message = ""
    monitor = None
    try:
        with subprocess.Popen(actual_command, cwd=ROOT, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
            try:
                monitor = ProcessMonitor(psutil.Process(process.pid))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            with monitor if monitor is not None else nullcontext():
                output_text = process.communicate()[0]
            return_code = process.returncode
    except OSError as error:
        output_text = ""
        error_message = str(error)
        return_code = 127
    record.total_wall_time_ns = time.monotonic_ns() - record.monotonic_start_ns
    peak_rss = monitor.peak_rss if monitor else 0
    peak_footprint = monitor.peak_footprint if monitor else 0
    samples = monitor.samples if monitor else 0
    record.peak_memory_bytes = peak_footprint or peak_rss
    (run_dir / "process.log").write_text(output_text)
    for line in output_text.splitlines():
        try:
            details = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(details, dict) or "stage_seconds" not in details:
            continue
        for stage_index, (name, seconds) in enumerate(details["stage_seconds"].items()):
            timing = record.stages.add()
            timing.stage.stage_id = name
            timing.stage.stage_index = stage_index
            timing.wall_time_ns = round(float(seconds) * 1e9)
        if "token_count" in details:
            record.metrics.add(name="text_tokens", value=details["token_count"], unit="tokens", source="runner")
    record.metrics.add(name="process_peak_rss", value=peak_rss, unit="bytes", source="psutil")
    record.metrics.add(name="process_peak_physical_footprint", value=peak_footprint, unit="bytes", source="psutil")
    record.metrics.add(name="memory_samples", value=samples, unit="count", source="psutil")
    record.metrics.add(name="swap_used_before", value=swap_before, unit="bytes", source="sysctl")
    record.metrics.add(name="swap_used_after", value=swap_bytes(), unit="bytes", source="sysctl")
    terminal = record.events.add()
    terminal.run_id = run_id
    terminal.request_id = request.request_id
    terminal.sequence = 1
    terminal.monotonic_time_ns = time.monotonic_ns()
    terminal.wall_time_unix_ns = time.time_ns()
    if return_code == 0 and output.is_file():
        artifact = artifact_from_path(output, types_pb2.ARTIFACT_KIND_IMAGE, "image/png", output)
        record.outputs.append(artifact)
        terminal.completed.SetInParent()
        if args.expect_output_sha256 and artifact.sha256.hex() != args.expect_output_sha256:
            return_code = 2
            error_message = f"Output SHA-256 differs: {artifact.sha256.hex()} != {args.expect_output_sha256}"
    else:
        return_code = return_code or 1
        error_message = error_message or f"Command exited {return_code}; output missing: {output}"
    if return_code != 0:
        terminal.failed.error.code = types_pb2.ERROR_CODE_BACKEND_FAILURE
        terminal.failed.error.retryability = types_pb2.RETRYABILITY_NO
        terminal.failed.error.message = error_message
        record.terminal_error.CopyFrom(terminal.failed.error)
    record_path = run_dir / "run.pb"
    record_path.write_bytes(record.SerializeToString(deterministic=True))
    write_proto_json(record, run_dir / "run.json")
    summary = {
        "run_id": run_id,
        "return_code": return_code,
        "seconds": record.total_wall_time_ns / 1e9,
        "peak_bytes": record.peak_memory_bytes,
        "output": str(output) if output.exists() else None,
        "output_sha256": record.outputs[0].sha256.hex() if record.outputs else None,
        "run_record": str(record_path),
        "error": error_message or None,
    }
    print(json.dumps(summary, indent=2))
    if return_code:
        raise RuntimeError(error_message)
    return record


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="onyx replay")
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--target", type=Path, default=ROOT / "config/targets/m5-max-36gb.json",
                        help="Device capabilities JSON recorded in the RunRecord")
    parser.add_argument("--output-name", default="output.png")
    parser.add_argument("--backend-version", required=True)
    parser.add_argument("--plan-id", default="")
    parser.add_argument("--numeric-profile", default="")
    parser.add_argument("--quantization-profile", default="")
    parser.add_argument("--resolved-source-revision", default="")
    parser.add_argument("--manifest-digest", default="")
    parser.add_argument("--rng-algorithm", default="unspecified")
    parser.add_argument("--rng-version", default="")
    parser.add_argument("--kernel-variant", action="append", default=[])
    parser.add_argument("--cache-state", type=lambda value: trace_pb2.CacheState.Value(f"CACHE_STATE_{value.upper()}"), default=trace_pb2.CACHE_STATE_COLD)
    parser.add_argument("--expect-output-sha256")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command:
        parser.error("A command is required after --")
    execute_case(args.case, args.output_root, command, args)


if __name__ == "__main__":
    main()
