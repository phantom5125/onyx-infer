import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_once(output_root, plan_id, case_path, native_bin=None, target=None, dit_prefetch=False):
    package = json.loads((ROOT / "artifacts/models/h3-w4/package-report.json").read_text())
    command = [
        sys.executable, "-m", "onyx.replay",
        "--case", str(case_path),
        "--output-root", str(output_root),
        "--backend-version", "onyx-native-cpp-metal",
        "--plan-id", plan_id,
        "--numeric-profile", "w4.bf16.fp32-sensitive",
        "--quantization-profile", "onyx-w4s64-linear-v1",
        "--resolved-source-revision", package["source_revision"],
        "--manifest-digest", package["semantic_digest"],
        "--rng-algorithm", "numpy.pcg64.standard_normal",
        "--rng-version", "2.5.4",
        "--cache-state", "unspecified",
        "--", sys.executable, "-m", "onyx.native",
        "--models", str(ROOT / "artifacts/models/h3-w4"),
        "--native-bin", str(native_bin or ROOT / ".build/onyx-native"),
        "--output", "{output}", "--prompt", "{prompt}",
        "--width", "{width}", "--height", "{height}", "--seed", "{seed}",
    ]
    if target is not None:
        command[3:3] = ["--target", str(target)]
    if dit_prefetch:
        command.append("--dit-prefetch")
    result = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    print(result.stdout, end="", flush=True)
    run = json.loads(result.stdout)
    record_path = Path(run["run_record"])
    record = json.loads(record_path.with_suffix(".json").read_text())
    run["stage_seconds"] = {
        stage["stage"]["stage_id"]: int(stage["wall_time_ns"]) / 1e9
        for stage in record["stages"]
    }
    for line in (record_path.parent / "process.log").read_text().splitlines():
        details = json.loads(line)
        if "stage_seconds" in details:
            run["backend"] = details
    return run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="run")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--plan-id")
    parser.add_argument("--case", type=Path, default=ROOT / "benchmarks/requests/teapot-1024.json")
    parser.add_argument("--native-bin", type=Path)
    parser.add_argument("--target", type=Path, help="Device capabilities JSON recorded in each RunRecord")
    parser.add_argument("--dit-prefetch", action="store_true")
    args = parser.parse_args()
    plan_id = args.plan_id or ("native.mpsgraph.q4.prefetch.v1" if args.dit_prefetch else "native.mpsgraph.q4.stream.v1")
    output_root = ROOT / "artifacts/benchmarks/onyx-native-h3"
    runs = [run_once(output_root, plan_id, args.case, args.native_bin, args.target, args.dit_prefetch)
            for _ in range(args.runs)]
    latest = {"run": runs[-1]}
    (output_root / "latest.json").write_text(json.dumps(latest, indent=2) + "\n")
    summary = {
        "label": args.label,
        "plan_id": plan_id,
        "runs": runs,
        "seconds_median": statistics.median(item["seconds"] for item in runs),
        "peak_bytes_max": max(item["peak_bytes"] for item in runs),
        "all_outputs_identical": len({item["output_sha256"] for item in runs}) == 1,
    }
    (output_root / f"{args.label}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "runs"}, indent=2))


if __name__ == "__main__":
    main()
