import argparse
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx


def percentile95(values):
    values = sorted(values)
    return values[min(len(values) - 1, int(len(values) * 0.95 + 0.999999) - 1)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--group", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mx.random.seed(42)
    activation = mx.random.normal((args.m, args.k), dtype=mx.float32).astype(mx.float16) * 0.2
    weights = mx.random.normal((args.n, args.k), dtype=mx.float16) * 0.2
    packed, scales, biases = mx.quantize(weights, group_size=args.group, bits=4)
    mx.eval(activation, packed, scales, biases)
    seconds = []
    for iteration in range(args.iterations + 1):
        begin = time.perf_counter_ns()
        result = mx.quantized_matmul(activation, packed, scales, biases, transpose=True, group_size=args.group, bits=4)
        mx.eval(result)
        if iteration:
            seconds.append((time.perf_counter_ns() - begin) / 1e9)
    flops = 2 * args.m * args.n * args.k
    median = statistics.median(seconds)
    report = {
        "operator": "mlx.quantized_matmul",
        "implementation": "h3-image-mlx environment",
        "mlx_version": mx.__version__,
        "device": mx.device_info()["device_name"],
        "shape": {"m": args.m, "n": args.n, "k": args.k, "group_size": args.group},
        "bits": 4,
        "activation_dtype": "float16",
        "quantization_mode": "mlx-affine-with-biases",
        "iterations": args.iterations,
        "wall_median_ms": median * 1000,
        "wall_p95_ms": percentile95(seconds) * 1000,
        "effective_tflops": flops / median / 1e12,
        "samples_ms": [value * 1000 for value in seconds],
        "comparison_scope": "Same logical W4A16 GEMM shape; different packed format and quantization numerics",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
