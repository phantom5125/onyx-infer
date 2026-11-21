import argparse
import hashlib
import json
import os
import struct
from pathlib import Path

import numpy as np

from onyx.safetensors import SafeTensorFile, component_index


MAGIC = b"ONYXPK01"
LINEAR_SUFFIXES = (
    ".attn.qkv_proj.weight",
    ".attn.out_proj.weight",
    ".mlp.fc1.weight",
    ".mlp.fc2.weight",
    ".q_proj.weight",
    ".k_proj.weight",
    ".v_proj.weight",
    ".o_proj.weight",
    ".gate_proj.weight",
    ".up_proj.weight",
    ".down_proj.weight",
)


def align_stream(stream, alignment=64):
    padding = (-stream.tell()) % alignment
    if padding:
        stream.write(b"\0" * padding)


def float32_view(values, dtype):
    if dtype == "BF16":
        bits = np.asarray(values, dtype=np.uint16).astype(np.uint32) << 16
        return bits.view(np.float32)
    return np.asarray(values, dtype=np.float32)


def hash_range(path, begin, size):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        stream.seek(begin)
        remaining = size
        while remaining:
            data = stream.read(min(8 * 1024 * 1024, remaining))
            if not data:
                raise OSError(f"Truncated tensor data: {path}")
            digest.update(data)
            remaining -= len(data)
    return digest.hexdigest()


class PackWriter:
    def __init__(self, output: Path, metadata):
        self.output = output
        self.metadata = metadata
        output.parent.mkdir(parents=True, exist_ok=True)
        self.body_path = output.with_suffix(output.suffix + ".body")
        self.body = self.body_path.open("wb")
        self.tensors = []

    def close(self):
        if not self.body.closed:
            self.body.close()

    def add_raw(self, source: SafeTensorFile, info):
        align_stream(self.body)
        begin = self.body.tell()
        digest = hashlib.sha256()
        with source.path.open("rb") as stream:
            stream.seek(info.absolute_begin)
            remaining = info.byte_size
            while remaining:
                data = stream.read(min(8 * 1024 * 1024, remaining))
                if not data:
                    raise OSError(f"Truncated tensor: {info.name}")
                self.body.write(data)
                digest.update(data)
                remaining -= len(data)
        end = self.body.tell()
        self.tensors.append({
            "name": info.name,
            "dtype": info.dtype,
            "shape": list(info.shape),
            "data_offsets": [begin, end],
            "source_dtype": info.dtype,
            "source_sha256": digest.hexdigest(),
        })

    def add_q4(self, source: SafeTensorFile, info, group_size=64, rows_per_chunk=16):
        if len(info.shape) != 2:
            raise ValueError(f"Q4 requires a rank-2 tensor: {info.name}")
        rows, columns = info.shape
        if columns % group_size or group_size % 2:
            raise ValueError(f"K={columns} must be divisible by even group size {group_size}: {info.name}")
        groups = columns // group_size
        values = source.memmap(info)
        align_stream(self.body)
        packed_begin = self.body.tell()
        scales = []
        squared_error = 0.0
        max_error = 0.0
        for row in range(0, rows, rows_per_chunk):
            block = float32_view(values[row:min(row + rows_per_chunk, rows)], info.dtype)
            current_rows = block.shape[0]
            grouped = block.reshape(current_rows, groups, group_size)
            scale = np.max(np.abs(grouped), axis=2, keepdims=True) / np.float32(7.0)
            safe_scale = np.where(scale != 0, scale, np.float32(1.0))
            quantized = np.clip(np.rint(grouped / safe_scale), -7, 7).astype(np.int8)
            restored = quantized.astype(np.float32) * scale
            error = restored - grouped
            squared_error += float(np.sum(error.astype(np.float64) ** 2))
            max_error = max(max_error, float(np.max(np.abs(error))))
            unsigned = (quantized + 8).astype(np.uint8).reshape(current_rows, columns // 2, 2)
            packed = unsigned[:, :, 0] | (unsigned[:, :, 1] << 4)
            self.body.write(packed.tobytes(order="C"))
            scales.append(scale.reshape(current_rows, groups).astype(np.float32))
        packed_end = self.body.tell()
        align_stream(self.body)
        scale_begin = self.body.tell()
        for block in scales:
            self.body.write(block.tobytes(order="C"))
        scale_end = self.body.tell()
        self.tensors.append({
            "name": info.name,
            "dtype": f"Q4S{group_size}",
            "shape": list(info.shape),
            "data_offsets": [packed_begin, packed_end],
            "scale_offsets": [scale_begin, scale_end],
            "scale_dtype": "F32",
            "packing": "unsigned-nibble(q+8); low=even-k; high=odd-k; row-major [N,K]",
            "source_dtype": info.dtype,
            "source_sha256": hash_range(source.path, info.absolute_begin, info.byte_size),
            "quantization_rmse": (squared_error / info.element_count) ** 0.5,
            "quantization_max_abs_error": max_error,
        })

    def finish(self):
        self.close()
        header = dict(self.metadata)
        header["format"] = "onyx.pack.v1"
        header["tensors"] = self.tensors
        encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
        partial = self.output.with_suffix(self.output.suffix + ".part")
        with partial.open("wb") as output:
            output.write(MAGIC)
            output.write(struct.pack("<Q", len(encoded)))
            output.write(encoded)
            with self.body_path.open("rb") as body:
                while data := body.read(8 * 1024 * 1024):
                    output.write(data)
        os.replace(partial, self.output)
        self.body_path.unlink()


class PackReader:
    def __init__(self, path: Path):
        self.path = Path(path)
        with self.path.open("rb") as stream:
            if stream.read(8) != MAGIC:
                raise ValueError(f"Not an Onyx pack: {path}")
            header_size = struct.unpack("<Q", stream.read(8))[0]
            if header_size > min(self.path.stat().st_size - 16, 256 * 1024 * 1024):
                raise ValueError(f"Invalid Onyx pack header size: {path}")
            self.header = json.loads(stream.read(header_size))
        if self.header.get("format") != "onyx.pack.v1":
            raise ValueError("Unsupported pack format")
        self.data_begin = 16 + header_size
        self.tensors = {item["name"]: item for item in self.header["tensors"]}

    def array(self, entry, field, dtype, shape=None):
        begin, end = entry[field]
        if shape is None:
            shape = entry["shape"]
        array = np.memmap(self.path, dtype=dtype, mode="r", offset=self.data_begin + begin, shape=tuple(shape), order="C")
        expected = array.nbytes
        if end - begin != expected:
            raise ValueError(f"Invalid {field} for {entry['name']}")
        return array

    def q4_arrays(self, name):
        entry = self.tensors[name]
        rows, columns = entry["shape"]
        group_size = int(entry["dtype"].removeprefix("Q4S"))
        packed = self.array(entry, "data_offsets", np.uint8, (rows, columns // 2))
        scales = self.array(entry, "scale_offsets", np.float32, (rows, columns // group_size))
        return entry, packed, scales


def selected_tensors(index, files, names, all_linear):
    by_name = {}
    for source, parsed in files.items():
        for info in parsed.tensors.values():
            by_name[info.name] = (parsed, info)
    if names:
        missing = set(names) - set(by_name)
        if missing:
            raise KeyError(f"Unknown tensors: {sorted(missing)}")
        selected = set(names)
    elif all_linear:
        selected = {name for name, (_, info) in by_name.items() if len(info.shape) == 2 and name.endswith(LINEAR_SUFFIXES)}
    else:
        raise ValueError("Select --tensor or --all-linear")
    ordered = list(index["weight_map"]) if all_linear else [name for name in index["weight_map"] if name in selected]
    return [(name, *by_name[name], name in selected) for name in ordered]


def quantize_component(source: Path, component: str, output: Path, names, all_linear=False, group_size=64):
    index, files = component_index(source, component, names or None)
    selected = selected_tensors(index, files, names, all_linear)
    if not selected:
        raise ValueError("No tensors selected")
    writer = PackWriter(output, {
        "source_component": component,
        "quantization": {
            "algorithm": "symmetric-groupwise-maxabs",
            "version": "onyx-q4-v1",
            "bits": 4,
            "signed_values": True,
            "group_size": group_size,
            "scale_dtype": "F32",
            "rounding": "nearest-even",
            "clipping": "[-7,7]",
        },
        "complete_component": bool(all_linear),
    })
    try:
        for position, (name, parsed, info, quantize) in enumerate(selected, 1):
            action = "Q4" if quantize else info.dtype
            print(f"[{position}/{len(selected)}] {action} {name} {info.shape}", flush=True)
            if quantize:
                writer.add_q4(parsed, info, group_size)
            else:
                writer.add_raw(parsed, info)
    finally:
        writer.close()
    writer.finish()
    report = {
        "output": str(output),
        "bytes": output.stat().st_size,
        "sha256": hash_range(output, 0, output.stat().st_size),
        "tensor_count": len(writer.tensors),
        "complete_component": bool(all_linear),
        "tensors": [{
            "name": item["name"],
            "shape": item["shape"],
            "dtype": item["dtype"],
            "source_sha256": item["source_sha256"],
            **({"rmse": item["quantization_rmse"], "max_abs_error": item["quantization_max_abs_error"]} if item["dtype"].startswith("Q4S") else {}),
        } for item in writer.tensors],
    }
    report_path = output.with_suffix(output.suffix + ".report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="onyx quantize")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--component", choices=("transformer", "text_encoder"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensor", action="append", default=[])
    parser.add_argument("--all-linear", action="store_true")
    parser.add_argument("--group-size", type=int, default=64)
    args = parser.parse_args(argv)
    report = quantize_component(args.source, args.component, args.output, args.tensor, args.all_linear, args.group_size)
    print(json.dumps({key: value for key, value in report.items() if key != "tensors"}, indent=2))


if __name__ == "__main__":
    main()
