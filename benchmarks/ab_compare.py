"""Interleaved A/B measurement for native generation builds and options.

Each arm names a native binary plus its own `onyx native-generate` options. The
arms run back to back in alternating order so thermal drift and position bias
fall on both sides, then medians and output hashes are compared.

    uv run python benchmarks/ab_compare.py --label r1 --rounds 3 --warmup 1 \
      --arm "base .build/baseline-bin/onyx-native --no-prefetch-weights" \
      --arm "cand .build/onyx-native"

Set ONYX_PROFILE=1 in the environment to collect the per-operation stage
profiles in the report.
"""

import argparse
import hashlib
import json
import os
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]


def case_request(case_path):
    if case_path:
        case = json.loads(Path(case_path).read_text())
        if "request" in case:
            return case["request"]
        # The text-rendering requests are flat: prompt, width, height, and seed.
        if "prompt" in case and "width" in case and "height" in case and "seed" in case:
            return {"prompt": case["prompt"], "size": {"width": case["width"], "height": case["height"]}, "seed": str(case["seed"])}
        raise SystemExit(f"{case_path} carries no request")
    return {"prompt": "A close-up product photograph of one bright yellow enamel teapot with a black arched handle and a round lid, standing on a blue wooden table. One small white ceramic cup sits immediately to its right. The background is plain warm beige. Soft morning window light falls from the left. Realistic enamel, clear silhouettes, no text.", "size": {"width": 1024, "height": 1024}, "seed": "42"}


class PeakSampler:
    """Track the peak resident set of a process tree while it runs."""

    def __init__(self, pid):
        self.process = psutil.Process(pid)
        self.peak = 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def sample(self):
        total = 0
        try:
            processes = [self.process, *self.process.children(recursive=True)]
        except psutil.Error:
            return  # The process exited; the last successful sample stands.
        for process in processes:
            try:
                total += process.memory_info().rss
            except psutil.Error:
                continue
        self.peak = max(self.peak, total)

    def run(self):
        while not self.stop.wait(0.05):
            self.sample()

    def finish(self):
        self.sample()
        self.stop.set()
        self.thread.join()
        return self.peak


def load_snapshot():
    """Return the one-minute load average and the busiest processes, to expose contended samples."""
    processes = sorted(psutil.process_iter(["pid", "name", "cmdline", "cpu_percent"]), key=lambda p: p.info["cpu_percent"] or 0.0)
    busiest = [{"pid": p.info["pid"], "name": p.info["name"], "cpu_percent": round(p.info["cpu_percent"] or 0.0, 1)} for p in processes[-3:]]
    return {"loadavg_1m": round(os.getloadavg()[0], 2), "busiest": busiest}


def native_stage_pids(exclude=()):
    """List PIDs of native stage processes, which are GPU work that would contend with a sample."""
    found = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        if process.info["pid"] in exclude or not process.info["cmdline"]:
            continue
        try:
            if process.info["cmdline"][0].endswith("/onyx-native") and process.info["cmdline"][1:2] and process.info["cmdline"][1] in ("dit", "text", "vae"):
                found.append(process.info["pid"])
        except psutil.Error:
            continue
    return found


def wait_for_idle(timeout):
    """Block until no native stage process is running, so each sample gets the GPU alone."""
    began = time.monotonic()
    while native_stage_pids():
        if time.monotonic() - began > timeout:
            break
        time.sleep(3)
    return time.monotonic() - began


def run_arm(arm, common, request, output, work):
    # An arm bound to another checkout runs that checkout's launcher, so a baseline is the
    # whole engine as shipped there: launcher, profile registry, and binary.
    command = [arm.get("python", sys.executable), "-m", "onyx.native", "--models", common.models, "--prompt", request["prompt"],
               "--output", str(output), "--width", str(request["size"]["width"]),
               "--height", str(request["size"]["height"]), "--seed", str(request["seed"]),
               "--profile", common.profile, "--native-bin", arm["binary"], *arm["args"], *common.args]
    for cache in common.cache_dirs:
        Path(cache).mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    for item in arm["env"]:
        key, _, value = item.partition("=")
        environment[key] = value
    psutil.cpu_percent(interval=None)  # Prime the per-process counters.
    idle_wait = wait_for_idle(common.idle_timeout)
    before = load_snapshot()
    began = time.monotonic()
    process = subprocess.Popen(command, cwd=arm.get("cwd", ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=environment)
    sampler = PeakSampler(process.pid)
    sampler.thread.start()
    stdout, stderr = process.communicate()
    peak = sampler.finish()
    wall = time.monotonic() - began
    foreign_after = native_stage_pids(exclude=(process.pid,))
    if process.returncode:
        raise RuntimeError(f"{arm['name']} failed:\n{stderr[-4000:]}")
    record = json.loads(stdout.strip().splitlines()[-1])
    stages = {}
    for name, log in record["stage_logs"].items():
        details = {}
        for line in log["stdout"].splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if item.get("stage") == name:
                    details = item
                elif "profile" in item:
                    details["profile_ops"] = item["profile"]
                    details["weight_read"] = item.get("weight_read")
                elif "foreground_weight_reads" in item:
                    details["foreground_weight_reads"] = item["foreground_weight_reads"]
        details.pop("output", None)
        details["stderr_tail"] = log["stderr"][-2000:]
        stages[name] = details
    digest = hashlib.sha256(Path(output).read_bytes()).hexdigest()
    return {"arm": arm["name"], "wall_seconds": wall, "total_seconds": record["total_seconds"],
            "stage_seconds": record["stage_seconds"], "token_count": record["token_count"],
            "peak_rss_bytes": peak, "png_sha256": digest, "png_bytes": Path(output).stat().st_size,
            "stages": stages, "idle_wait_seconds": round(idle_wait, 1),
            "foreign_native_after": foreign_after, "load_before": before, "load_after": load_snapshot()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", required=True)
    parser.add_argument("--rounds", type=int, default=3, help="Measured runs per arm")
    parser.add_argument("--warmup", type=int, default=1, help="Unmeasured runs per arm before the first round")
    parser.add_argument("--arm", action="append", required=True, metavar="'NAME BIN [ARG...]'",
                        help="One quoted arm: a name, a native binary, and native-generate options; repeat per arm")
    parser.add_argument("--env", action="append", default=[], metavar="NAME=KEY=VALUE",
                        help="Environment variable for one arm, for example cand=ONYX_PROFILE=1")
    parser.add_argument("--arm-root", action="append", default=[], metavar="NAME=PATH",
                        help="Checkout whose launcher and profile registry drive one arm; repeatable")
    parser.add_argument("--native-arg", action="append", default=[], help="native-generate option shared by every arm")
    parser.add_argument("--models", default=str(ROOT / "artifacts/models/h3-w4"))
    parser.add_argument("--profile", default="max-quality")
    parser.add_argument("--case", type=Path, help="Request JSON with the prompt, size, and seed")
    parser.add_argument("--cache-dir", action="append", default=[],
                        help="Directory to create before each run, for modulation caches")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/benchmarks/ab")
    parser.add_argument("--keep-outputs", type=Path, help="Directory to copy each measured PNG into, named by arm and round")
    parser.add_argument("--idle-timeout", type=int, default=1800,
                        help="Seconds to wait for other native stage processes before running a sample anyway")
    args = parser.parse_args()

    arms = []
    for items in args.arm:
        items = shlex.split(items)
        if len(items) < 2:
            parser.error(f"--arm needs a name and a binary: {items}")
        arms.append({"name": items[0], "binary": str(Path(items[1]).resolve()), "args": items[2:], "env": []})
    names = [arm["name"] for arm in arms]
    if len(set(names)) != len(names):
        parser.error("Arm names must be unique")
    for item in args.env:
        target, _, assignment = item.partition("=")
        key, _, value = assignment.partition("=")
        if not target or not key or target not in names:
            parser.error(f"--env must be NAME=KEY=VALUE with a known arm: {item}")
        arms[names.index(target)]["env"].append(f"{key}={value}")
    for item in args.arm_root:
        target, _, checkout = item.partition("=")
        if not target or not checkout or target not in names:
            parser.error(f"--arm-root must be NAME=PATH with a known arm: {item}")
        root = Path(checkout).resolve()
        if not (root / ".venv/bin/python").is_file() or not (root / "src/onyx/native.py").is_file():
            parser.error(f"{root} is not a checkout with a synced environment")
        arms[names.index(target)].update(root=str(root), python=str(root / ".venv/bin/python"), cwd=str(root))
    request = case_request(args.case)
    common = argparse.Namespace(models=str(Path(args.models).resolve()), profile=args.profile, args=args.native_arg,
                                cache_dirs=args.cache_dir, idle_timeout=args.idle_timeout)

    runs = []
    with tempfile.TemporaryDirectory(prefix="onyx-ab-") as temporary:
        for warm in range(args.warmup):
            for arm in arms:
                run_arm(arm, common, request, Path(temporary) / f"{arm['name']}.png", Path(temporary))
        for round_index in range(args.rounds):
            order = arms[round_index % len(arms):] + arms[:round_index % len(arms)]
            for arm in order:
                print(f"round {round_index + 1}/{args.rounds}: {arm['name']} ...", flush=True)
                record = run_arm(arm, common, request, Path(temporary) / f"{arm['name']}.png", Path(temporary))
                record["round"] = round_index
                runs.append(record)
                if args.keep_outputs:
                    args.keep_outputs.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(Path(temporary) / f"{arm['name']}.png",
                                    args.keep_outputs / f"{args.label}-{arm['name']}-{round_index}.png")
                stages = " / ".join(f"{name} {seconds:.2f}" for name, seconds in record["stage_seconds"].items())
                print(f"  total {record['total_seconds']:.2f} s  ({stages})  sha {record['png_sha256'][:12]}", flush=True)

    summary = {}
    for arm in arms:
        samples = [record for record in runs if record["arm"] == arm["name"]]
        totals = [record["total_seconds"] for record in samples]
        entry = {"samples_seconds": totals, "seconds_median": statistics.median(totals),
                 "seconds_min": min(totals), "seconds_max": max(totals),
                 "stage_seconds_median": {name: statistics.median(record["stage_seconds"][name] for record in samples)
                                          for name in samples[0]["stage_seconds"]},
                 "peak_rss_bytes_max": max(record["peak_rss_bytes"] for record in samples),
                 "contended_rounds": [record["round"] for record in samples if record.get("foreign_native_after")],
                 "binary": arm["binary"], "args": arm["args"], "env": arm["env"], "launcher_root": arm.get("root"),
                 "output_sha256": sorted({record["png_sha256"] for record in samples})}
        entry["all_outputs_identical"] = len(entry["output_sha256"]) == 1
        summary[arm["name"]] = entry
    reference = summary[names[0]]["seconds_median"]
    for name in names[1:]:
        summary[name]["speedup_vs_first_arm"] = reference / summary[name]["seconds_median"]

    report = {"protocol": "onyx.ab-compare.v1", "label": args.label, "profile": args.profile,
              "request": {key: request[key] for key in ("prompt", "size", "seed")},
              "method": "Same machine and W4 package; arms alternate in rotating order after per-arm warmups",
              "warmups_per_arm": args.warmup, "rounds": args.rounds, "common_args": args.native_arg,
              "profile_enabled": os.environ.get("ONYX_PROFILE") is not None,
              "summary": summary, "runs": runs}
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"{args.label}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    for name in names:
        entry = summary[name]
        stages = " / ".join(f"{key} {value:.2f}" for key, value in entry["stage_seconds_median"].items())
        speed = f"  {entry['speedup_vs_first_arm']:.3f}x vs {names[0]}" if name != names[0] else ""
        print(f"{name}: median {entry['seconds_median']:.2f} s ({stages})  "
              f"peak {entry['peak_rss_bytes_max'] / 2**30:.2f} GiB  identical={entry['all_outputs_identical']}{speed}")
    print(f"report: {path}")


if __name__ == "__main__":
    main()
