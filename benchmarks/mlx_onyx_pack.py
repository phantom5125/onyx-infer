import argparse
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from onyx.quantize import PackReader


def to_mlx_pack(packed, scales):
    rows, columns = packed.shape[0], packed.shape[1] * 2
    values = np.stack((packed & 15, packed >> 4), axis=2).reshape(rows, columns, 1)
    values = values.reshape(rows, columns // 8, 8).astype(np.uint32)
    shifts = (np.arange(8, dtype=np.uint32) * 4).reshape(1, 1, 8)
    packed32 = np.bitwise_or.reduce(values << shifts, axis=2)
    return mx.array(packed32), mx.array(np.asarray(scales).astype(np.float16)), mx.array(np.asarray(scales * -8).astype(np.float16))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--tensor", required=True)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pack = PackReader(args.pack)
    entry, packed, scales = pack.q4_arrays(args.tensor)
    fixture = json.loads((args.fixture / "fixture.json").read_text())
    shape = fixture["shape"]
    activation = mx.array(np.fromfile(args.fixture / "input.f16", dtype=np.float16).reshape(shape["m"], shape["k"]))
    wq, scale, bias = to_mlx_pack(packed, scales)
    mx.eval(activation, wq, scale, bias)
    seconds = []
    for iteration in range(args.iterations + 1):
        begin = time.perf_counter_ns()
        result = mx.quantized_matmul(activation, wq, scale, bias, transpose=True, group_size=shape["group_size"], bits=4)
        mx.eval(result)
        if iteration:
            seconds.append((time.perf_counter_ns() - begin) / 1e9)
    median = statistics.median(seconds)
    flops = 2 * shape["m"] * shape["n"] * shape["k"]
    report = {
        "operator": "mlx.quantized_matmul",
        "source": "Identical Onyx Q4 tensor repacked to MLX affine uint32 format",
        "tensor": args.tensor,
        "pack_sha256": fixture["pack_sha256"],
        "shape": shape,
        "mlx_version": mx.__version__,
        "iterations": args.iterations,
        "wall_median_ms": median * 1000,
        "wall_p95_ms": sorted(seconds)[min(len(seconds) - 1, int(len(seconds) * 0.95 + 0.999999) - 1)] * 1000,
        "effective_tflops": flops / median / 1e12,
        "samples_ms": [value * 1000 for value in seconds],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
