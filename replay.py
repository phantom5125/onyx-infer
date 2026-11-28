"""Run the real native launcher, recording each stage's rusage and DiT output SHA-256."""
import argparse
import contextlib
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import time
import threading
from unittest.mock import patch

import psutil
from onyx import native

ROOT = Path.cwd()
OUT = ROOT / ".build/adaln-evidence"
run = subprocess.run


def digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def other_native():
    result = []
    own = {process.pid for process in psutil.Process().children(recursive=True)}
    for process in psutil.process_iter(["pid", "cmdline"]):
        args = process.info["cmdline"] or []
        if process.pid not in own and len(args) >= 2 and Path(args[0]).name == "onyx-native" and args[1] in {"text", "dit", "vae"}:
            result.append(process.info["pid"])
    return result


def generate(resolution, build, mode, output, profile=False, reference=None, abort_on_contention=False):
    binary = (ROOT / ".build" / build / "onyx-native").resolve(strict=True)
    build_digest = digest(binary)
    request = json.loads((ROOT / f"benchmarks/requests/teapot-{resolution}.json").read_text())["request"]
    if reference is not None:
        request["prompt"] = "Make an ink outline of the teapot from <Picture 1> on white paper."
    stages = {}
    cache = OUT / "modulation" / build_digest
    cache.mkdir(parents=True, exist_ok=True)
    launch_args = [
        "--models", str(ROOT / "artifacts/models/h3-w4"),
        "--prompt", request["prompt"], "--seed", request["seed"],
        "--width", str(resolution), "--height", str(resolution),
        "--native-bin", str(binary), "--output", str(output), "--modulation-cache", str(cache),
        "--profile", "max-quality", "--dit-cache-mib", "0", "--dit-prefetch",
    ]

    if reference is not None:
        launch_args.extend(["--reference", str(reference.resolve(strict=True))])

    def measured_stage(command, *args, **kwargs):
        if command[0] != str(binary):
            return run(command, *args, **kwargs)
        stage = command[1]
        if abort_on_contention and foreign_during:
            raise RuntimeError(f"Overlapping native stages during replay: {sorted(foreign_during)}")
        environment = kwargs["env"]
        if stage == "dit":
            environment["ONYX_DIT_METAL_GATE"] = mode
            if reference is not None:
                environment["ONYX_DIT_INDEXED_MODULATION"] = "1"
            if profile:
                environment["ONYX_PROFILE"] = "1"
        environment["ONYX_VAE_BATCH"] = "25"
        began = time.monotonic()
        result = run(["/usr/bin/time", "-l", *command], *args, **kwargs)
        if abort_on_contention and foreign_during:
            raise RuntimeError(f"Overlapping native stages during replay: {sorted(foreign_during)}")
        details = {
            "wall_seconds": time.monotonic() - began,
            "peak_footprint_bytes": int(re.search(r"(?m)^\s*(\d+)\s+peak memory footprint", result.stderr)[1]),
            "peak_rss_bytes": int(re.search(r"(?m)^\s*(\d+)\s+maximum resident set size", result.stderr)[1]),
        }
        if stage == "dit" and result.returncode == 0:
            details["latent_sha256"] = digest(command[command.index("--output") + 1])
        stages[stage] = details
        return result

    clean_env = {key: val for key, val in os.environ.items() if not key.startswith("ONYX_")}
    began = datetime.datetime.now(datetime.timezone.utc).isoformat()
    foreign_before = other_native()
    if foreign_before:
        raise RuntimeError(f"Competing native stages before launch: {foreign_before}")
    swap_before = psutil.swap_memory().used
    stop_monitor = threading.Event()
    foreign_during = set()

    def monitor():
        while not stop_monitor.wait(2):
            foreign_during.update(other_native())
            if abort_on_contention and foreign_during:
                # Signal only the native descendant of this benchmark process.
                for child in psutil.Process().children(recursive=True):
                    try:
                        argv = child.cmdline()
                        if argv and argv[0] == str(binary) and argv[1:2] in (["text"], ["dit"], ["vae"]):
                            child.terminate()
                    except psutil.Error:
                        pass

    thread = threading.Thread(target=monitor)
    thread.start()
    try:
        with patch.dict(os.environ, clean_env, clear=True), patch.object(native.subprocess, "run", measured_stage):
            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                native.main(launch_args)
    finally:
        stop_monitor.set()
        thread.join()
    record = json.loads(stdout.getvalue())
    return {
        "started_at": began,
        "build": build,
        "binary_sha256": build_digest,
        "metallib_sha256": digest(binary.with_name("onyx_q4.metallib")),
        "metal_gate_mode": mode,
        "profiled": profile,
        "request_resolution": resolution,
        "png_sha256": digest(output),
        "stage_measurements": stages,
        "swap_before_bytes": swap_before,
        "swap_after_bytes": psutil.swap_memory().used,
        "foreign_native_during": sorted(foreign_during),
        "foreign_native_after": other_native(),
        **record,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("resolution", type=int)
    parser.add_argument("arms", nargs="+", help="Each arm is NAME:BUILD:MODE")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--require-exact", action="store_true")
    parser.add_argument("--warm-cache", action="store_true")
    parser.add_argument("--abort-on-contention", action="store_true")
    parser.add_argument("--reference", type=Path)
    arguments = parser.parse_args()
    directory = OUT / arguments.name
    directory.mkdir(parents=True, exist_ok=True)
    expected = None
    for index, arm in enumerate(arguments.arms, 1):
        name, build, mode = arm.split(":")
        sample = directory / f"{index:02d}-{name}.json"
        if sample.exists():
            print(f"Already recorded {sample}", flush=True)
            record = json.loads(sample.read_text())
        else:
            print(f"Started {index}/{len(arguments.arms)} {name}", flush=True)
            record = generate(arguments.resolution, build, mode, sample.with_suffix(".png"),
                              arguments.profile, arguments.reference, arguments.abort_on_contention)
            record["arm"] = name
            sample.write_text(json.dumps(record, indent=2) + "\n")
        dit = record["stage_measurements"]["dit"]
        hashes = (dit["latent_sha256"], record["png_sha256"])
        if expected is None:
            expected = hashes
        if arguments.require_exact and hashes != expected:
            raise RuntimeError(f"Output hash mismatch in {sample}")
        if arguments.warm_cache and record["modulation_cache"]["status"] != "hit":
            raise RuntimeError(f"Missing warm modulation cache in {sample}")
        print(f"Finished {name}: {dit['wall_seconds']:.3f}s DiT, {record['total_seconds']:.3f}s total, "
              f"{dit['peak_footprint_bytes']/2**30:.3f} GiB peak, {record['png_sha256'][:12]}", flush=True)


if __name__ == "__main__":
    main()
