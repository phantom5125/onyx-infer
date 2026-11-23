"""Turn an interleaved A/B report into an onyx.native-optimization.v1 round record.

    uv run python benchmarks/ab_record.py --report artifacts/benchmarks/ab/r2.json \
      --round 2 --change "..." --base-commit 75d73d7 --baseline r1 --candidate r2 \
      --baseline-binary .build/round1-bin/onyx-native --candidate-binary .build/onyx-native \
      --output benchmarks/results/campaign-20261123/round2-modulation-gather.json
"""

import argparse
import hashlib
import json
import platform
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = {"h3-teapot-1024-seed42-v1": "1024×1024"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arm(report, name, binary, env):
    summary = report["summary"][name]
    runs = []
    for record in report["runs"]:
        if record["arm"] != name:
            continue
        dit = record["stages"]["dit"]
        runs.append({
            "seconds": record["total_seconds"],
            "stage_seconds": record["stage_seconds"],
            "peak_rss_bytes": record["peak_rss_bytes"],
            "png_sha256": record["png_sha256"],
            "dit_step_seconds": [float(value) for _, value in re.findall(r"step (\d)/6, ([\d.]+) s", dit["stderr_tail"])],
            "dit_native_seconds": dit["seconds"],
            "dit_prefetch_wait_seconds": dit.get("prefetch_wait_seconds", 0.0),
            "dit_read_bytes": dit.get("weight_cache", {}).get("read_bytes"),
        })
    return {
        "binary": binary,
        "native_args": summary["args"],
        "environment": env,
        "samples_seconds": summary["samples_seconds"],
        "seconds_median": summary["seconds_median"],
        "stage_seconds_median": summary["stage_seconds_median"],
        "peak_bytes_max": summary["peak_rss_bytes_max"],
        "output_sha256": summary["output_sha256"],
        "all_outputs_identical": summary["all_outputs_identical"],
        "runs": runs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--change", required=True)
    parser.add_argument("--base-commit", required=True)
    parser.add_argument("--baseline", required=True, help="Arm name of the baseline")
    parser.add_argument("--candidate", required=True, help="Arm name of the candidate")
    parser.add_argument("--baseline-binary", type=Path, required=True)
    parser.add_argument("--candidate-binary", type=Path, required=True)
    parser.add_argument("--baseline-env", default=[], help="KEY=VALUE applied to the baseline arm; repeatable")
    parser.add_argument("--candidate-env", default=[], help="KEY=VALUE applied to the candidate arm; repeatable")
    parser.add_argument("--decision", default="accepted", choices=["accepted", "rejected", "inconclusive"])
    parser.add_argument("--note", action="append", default=[], help="Free-text caveat; repeatable")
    parser.add_argument("--attribution", type=Path, help="JSON file with free-form attribution facts")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.report.read_text())
    baseline = arm(report, args.baseline, str(args.baseline_binary), args.baseline_env)
    candidate = arm(report, args.candidate, str(args.candidate_binary), args.candidate_env)
    request = report["request"]
    case = next((key for key, label in CASES.items() if label == f"{request['size']['width']}×{request['size']['height']}"
                 and str(request["seed"]) == "42"), "custom")
    target = json.loads((ROOT / "config/targets/m5-max-36gb.json").read_text())
    record = {
        "protocol": "onyx.native-optimization.v1",
        "campaign": "local-single-image-20261123",
        "round": args.round,
        "change": args.change,
        "base_commit": args.base_commit,
        "case": case,
        "profile": report["profile"],
        "device": {
            "model_identifier": target["model_identifier"],
            "chip": target["chip"],
            "gpu_cores": target["gpu_cores"],
            "physical_memory_bytes": target["physical_memory_bytes"],
            "os_version": platform.mac_ver()[0],
        },
        "native_binary": {
            "baseline_sha256": sha256(args.baseline_binary),
            "candidate_sha256": sha256(args.candidate_binary),
            "same_binary": sha256(args.baseline_binary) == sha256(args.candidate_binary),
        },
        "method": report["method"] + ". Arm order: " + " | ".join(
            f"round {index + 1}: " + ", ".join(record["arm"] for record in report["runs"] if record["round"] == index)
            for index in range(report["rounds"])),
        "baseline": baseline,
        "candidate": candidate,
        "speedup_vs_baseline": baseline["seconds_median"] / candidate["seconds_median"],
        "latency_reduction": 1 - candidate["seconds_median"] / baseline["seconds_median"],
        "output_sha256": candidate["output_sha256"],
        "outputs_match_baseline": sorted(candidate["output_sha256"]) == sorted(baseline["output_sha256"]),
        "decision": args.decision,
        "notes": args.note,
    }
    if args.attribution:
        record["attribution"] = json.loads(args.attribution.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print(f"{args.output}: baseline {baseline['seconds_median']:.3f} s, candidate {candidate['seconds_median']:.3f} s, "
          f"{record['speedup_vs_baseline']:.4f}x, outputs match baseline: {record['outputs_match_baseline']}")


if __name__ == "__main__":
    main()
