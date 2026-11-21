import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAIRS = [
    ("h3-linear-m1024-n5376-k5376", "artifacts/benchmarks/onyx/q4-linear-1024x5376x5376.json", "artifacts/benchmarks/mlx/q4-linear-1024x5376x5376.json", False),
    ("h3-qkv-m1024-n21504-k5376", "artifacts/benchmarks/onyx/q4-qkv-1024x21504x5376.json", "artifacts/benchmarks/mlx/q4-qkv-1024x21504x5376.json", False),
    ("h3-text-q0-qproj-m69", "artifacts/benchmarks/onyx/h3-text-q0-qproj-m69.json", "artifacts/benchmarks/mlx/h3-text-q0-qproj-m69.json", True),
]


def main():
    comparisons = []
    for name, onyx_path, mlx_path, same_weights in PAIRS:
        onyx = json.loads((ROOT / onyx_path).read_text())
        mlx = json.loads((ROOT / mlx_path).read_text())
        onyx_time = onyx["wall_median_ms"]
        mlx_time = mlx["wall_median_ms"]
        comparisons.append({
            "case_id": name,
            "shape": onyx["shape"],
            "same_source_weights": same_weights,
            "quantization_format_note": "Identical Onyx tensor repacked for MLX" if same_weights else "Independent affine quantization; performance-only comparison",
            "onyx": {
                "implementation": "onyx_q4_gemm_fp16acc32 tile16",
                "wall_median_ms": onyx_time,
                "wall_p95_ms": onyx["wall_p95_ms"],
                "gpu_median_ms": onyx["gpu_median_ms"],
                "correctness_scope": f"{onyx['checked_outputs']} outputs checked against CPU dequantized FP32 accumulation",
                "correct": onyx["correct"],
                "max_abs_error": onyx["max_abs_error"],
                "samples_ms": onyx.get("wall_samples_ms", []),
                "source": onyx_path,
            },
            "mlx": {
                "implementation": "MLX 0.32.4 quantized_matmul",
                "wall_median_ms": mlx_time,
                "wall_p95_ms": mlx["wall_p95_ms"],
                "samples_ms": mlx["samples_ms"],
                "source": mlx_path,
            },
            "onyx_over_mlx_time_ratio": onyx_time / mlx_time,
        })
    report = {
        "protocol": "onyx.operator-comparison.v1",
        "target_id": "mac17-14-m5-max-36gb",
        "device": "Apple M5 Max 32-core GPU, 36 GB",
        "primary_metric": "wall_median_ms after one warmup; process-local steady state",
        "comparisons": comparisons,
        "scope_warning": "Operator results are not end-to-end image-generation results and do not demonstrate final image quality.",
    }
    output = ROOT / "benchmarks/results/m5-max-36gb-q4-operator-v1.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    for item in comparisons:
        print(f"{item['case_id']}: Onyx/MLX={item['onyx_over_mlx_time_ratio']:.3f}x")


if __name__ == "__main__":
    main()
