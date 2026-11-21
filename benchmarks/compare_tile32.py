import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CASES = [
    ("h3-text-q0-qproj-m69", "h3-text-q0-qproj-m69"),
    ("h3-qkv-m1024-n21504-k5376", "q4-qkv-1024x21504x5376"),
]


def main():
    results = []
    for case, stem in CASES:
        baseline_path = ROOT / f"artifacts/benchmarks/onyx/{stem}.json"
        candidate_path = ROOT / f"artifacts/benchmarks/onyx/{stem}-tile32.json"
        baseline = json.loads(baseline_path.read_text())
        candidate = json.loads(candidate_path.read_text())
        if not baseline["correct"] or not candidate["correct"]:
            raise RuntimeError(f"Numerical gate failed for {case}")
        results.append({
            "case_id": case,
            "shape": candidate["shape"],
            "baseline_variant": "tile16",
            "candidate_variant": "tile32",
            "baseline_wall_median_ms": baseline["wall_median_ms"],
            "candidate_wall_median_ms": candidate["wall_median_ms"],
            "speedup": baseline["wall_median_ms"] / candidate["wall_median_ms"],
            "candidate_max_abs_error": candidate["max_abs_error"],
            "baseline_samples_ms": baseline["wall_samples_ms"],
            "candidate_samples_ms": candidate["wall_samples_ms"],
            "baseline_source": str(baseline_path.relative_to(ROOT)),
            "candidate_source": str(candidate_path.relative_to(ROOT)),
        })
    report = {
        "protocol": "onyx.optimization-comparison.v1",
        "target_id": "mac17-14-m5-max-36gb",
        "primary_metric": "wall_median_ms",
        "results": results,
        "scope": "Same Q4 tensors, shapes and numerical checks; only Metal tiling variant changes",
    }
    output = ROOT / "benchmarks/results/m5-max-36gb-q4-tile32-v1.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    for result in results:
        print(f"{result['case_id']}: tile32 speedup {result['speedup']:.3f}x")


if __name__ == "__main__":
    main()
