import json
import subprocess
import sys
from pathlib import Path

from onyx.trace.v1 import trace_pb2


ROOT = Path(__file__).resolve().parents[1]


def main():
    community = json.loads((ROOT / "artifacts/benchmarks/community-h3/summary.json").read_text())
    reference_run = json.loads((ROOT / "artifacts/benchmarks/onyx-reference-h3/latest.json").read_text())["run"]
    record = trace_pb2.RunRecord.FromString(Path(reference_run["run_record"]).read_bytes())
    metrics_path = ROOT / "artifacts/benchmarks/h3-teapot-1024-image-metrics.json"
    subprocess.run([
        sys.executable, str(ROOT / "benchmarks/image_metrics.py"),
        "--reference", community["runs"][-1]["output"], "--candidate", reference_run["output"],
        "--output", str(metrics_path),
    ], check=True)
    image_metrics = json.loads(metrics_path.read_text())
    community_median = community["seconds_median"]
    reference_seconds = reference_run["seconds"]
    report = {
        "protocol": "onyx.generation-comparison.v1",
        "case_id": "h3-teapot-1024-seed42-v1",
        "target_id": "mac17-14-m5-max-36gb",
        "community": {
            "implementation": "Anionex/h3-image-mlx W8A8 resident",
            "revision": community["implementation_revision"],
            "seconds_median": community_median,
            "seconds_p95": community["seconds_p95"],
            "measured_runs": 3,
            "peak_bytes_max": community["peak_bytes_max"],
            "all_outputs_identical": community["all_outputs_identical"],
        },
        "onyx_reference": {
            "implementation": "Independent Onyx MLX W8A16 reference; not native C++ Metal",
            "seconds": reference_seconds,
            "peak_bytes": reference_run["peak_bytes"],
            "stages": {stage.stage.stage_id: stage.wall_time_ns / 1e9 for stage in record.stages},
            "run_record": reference_run["run_record"],
        },
        "reference_over_community_time_ratio": reference_seconds / community_median,
        "image_similarity": image_metrics,
        "scope": "Same prompt, seed, dimensions and source weights. Community uses W8A8; reference uses W8A16. This is not a native Onyx W4 performance result or an absolute quality evaluation.",
        "native_onyx_w4_status": "not included",
    }
    w4_latest = ROOT / "artifacts/benchmarks/onyx-w4-reference-h3/latest.json"
    if w4_latest.exists():
        w4_run = json.loads(w4_latest.read_text())["run"]
        w4_record = trace_pb2.RunRecord.FromString(Path(w4_run["run_record"]).read_bytes())
        w4_metrics_path = ROOT / "artifacts/benchmarks/h3-teapot-1024-w4-image-metrics.json"
        subprocess.run([
            sys.executable, str(ROOT / "benchmarks/image_metrics.py"),
            "--reference", community["runs"][-1]["output"], "--candidate", w4_run["output"],
            "--output", str(w4_metrics_path),
        ], check=True)
        report["onyx_w4_reference"] = {
            "implementation": "Onyx original-checkpoint W4BF16 MLX reference; not native C++ Metal",
            "seconds": w4_run["seconds"],
            "peak_bytes": w4_run["peak_bytes"],
            "stages": {stage.stage.stage_id: stage.wall_time_ns / 1e9 for stage in w4_record.stages},
            "run_record": w4_run["run_record"],
        }
        report["w4_reference_over_community_time_ratio"] = w4_run["seconds"] / community_median
        report["w4_image_similarity"] = json.loads(w4_metrics_path.read_text())
        report["native_onyx_w4_status"] = "not included; W4 result is the MLX reference backend"
    native_latest = ROOT / "artifacts/benchmarks/onyx-native-h3/round2-direct-pread.json"
    if native_latest.exists():
        native_run = json.loads(native_latest.read_text())["runs"][-1]
        native_record = trace_pb2.RunRecord.FromString(Path(native_run["run_record"]).read_bytes())
        native_metrics_path = ROOT / "artifacts/benchmarks/h3-teapot-1024-native-image-metrics.json"
        subprocess.run([
            sys.executable, str(ROOT / "benchmarks/image_metrics.py"),
            "--reference", community["runs"][-1]["output"], "--candidate", native_run["output"],
            "--output", str(native_metrics_path),
        ], check=True)
        report["onyx_native"] = {
            "implementation": "Onyx C++ Metal with MPSGraph and native Onyx Q4 pack",
            "seconds": native_run["seconds"],
            "peak_bytes": native_run["peak_bytes"],
            "stages": {stage.stage.stage_id: stage.wall_time_ns / 1e9 for stage in native_record.stages},
            "run_record": native_run["run_record"],
        }
        report["native_over_community_time_ratio"] = native_run["seconds"] / community_median
        report["native_image_similarity"] = json.loads(native_metrics_path.read_text())
        report["native_onyx_w4_status"] = "completed-native-cpp-metal"
    output = ROOT / "benchmarks/results/h3-teapot-1024-comparison-v1.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
