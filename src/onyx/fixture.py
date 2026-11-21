import argparse
import json
from pathlib import Path

import numpy as np

from onyx.artifacts import sha256_file
from onyx.quantize import PackReader


def export_fixture(pack_path: Path, tensor: str, output: Path, rows: int, seed=42):
    pack = PackReader(pack_path)
    entry, packed, scales = pack.q4_arrays(tensor)
    _, columns = entry["shape"]
    group_size = int(entry["dtype"].removeprefix("Q4S"))
    rng = np.random.default_rng(seed)
    activation = (rng.standard_normal((rows, columns), dtype=np.float32) * 0.2).astype(np.float16)
    output.mkdir(parents=True, exist_ok=True)
    activation.tofile(output / "input.f16")
    with (output / "weights.q4").open("wb") as stream:
        for row in range(0, packed.shape[0], 64):
            stream.write(np.asarray(packed[row:row + 64]).tobytes(order="C"))
    with (output / "scales.f32").open("wb") as stream:
        for row in range(0, scales.shape[0], 64):
            stream.write(np.asarray(scales[row:row + 64]).tobytes(order="C"))
    metadata = {
        "protocol": "onyx.metal.fixture.v1",
        "tensor": tensor,
        "shape": {"m": rows, "n": entry["shape"][0], "k": columns, "group_size": group_size},
        "seed": seed,
        "pack_sha256": sha256_file(pack_path),
        "source_tensor_sha256": entry["source_sha256"],
        "activation": "NumPy PCG64 standard normal FP32, multiplied by 0.2, then FP16",
    }
    (output / "fixture.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(prog="onyx export-metal-fixture")
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--tensor", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    export_fixture(args.pack, args.tensor, args.output, args.rows, args.seed)


if __name__ == "__main__":
    main()
