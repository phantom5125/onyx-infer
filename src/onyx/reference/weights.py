import json
import os
import struct
from pathlib import Path

import mlx.core as mx
import numpy as np

from onyx.quantize import PackReader
from onyx.safetensors import SafeTensorFile


DTYPES = {"I8": "i1", "U8": "u1", "F16": "<f2", "BF16": "<u2", "F32": "<f4", "I64": "<i8"}
_ROTATION = None


def rotation():
    global _ROTATION
    if _ROTATION is None:
        h4 = np.ones((4, 4), dtype=np.float32)
        np.fill_diagonal(h4[:, ::-1], -1)
        h256 = np.kron(np.kron(h4, h4), np.kron(h4, h4)) / 16
        _ROTATION = mx.array(h256)
    return _ROTATION


def rotate(x):
    shape = x.shape
    return (x.astype(mx.float32).reshape(-1, 256) @ rotation()).reshape(shape)


class Weights:
    def __init__(self, path, resident=False, raw_source=None, raw_component="transformer"):
        self.path = Path(path)
        if self.path.suffix == ".onxpack":
            pack = PackReader(self.path)
            self.start = pack.data_begin
            self.header = pack.tensors
        else:
            with self.path.open("rb") as stream:
                (length,) = struct.unpack("<Q", stream.read(8))
                self.start = 8 + length
                self.header = json.loads(stream.read(length))
        self.file = self.path.open("rb")
        self.resident = resident
        self.cache = {}
        self.raw_source = Path(raw_source) if raw_source else None
        self.raw_component = raw_component
        self.raw_files = {}
        self.raw_index = None
        if self.raw_source:
            directory = self.raw_source / raw_component
            self.raw_index = json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"]

    def close(self):
        self.cache.clear()
        self.file.close()

    def release(self):
        self.cache.clear()

    def read(self, key, rows=None):
        cache_key = (key, None if rows is None else tuple(rows))
        if cache_key in self.cache:
            return self.cache[cache_key]
        spec = self.header[key]
        begin, end = spec["data_offsets"]
        shape = spec["shape"]
        quantized = spec["dtype"].startswith("Q4S")
        if quantized and rows is not None:
            raise ValueError("Q4 row slicing is only available through linear projection")
        if quantized:
            shape = [shape[0], shape[1] // 2]
        if rows is None:
            data = os.pread(self.file.fileno(), end - begin, self.start + begin)
        else:
            stride = (end - begin) // shape[0]
            data = b"".join(os.pread(self.file.fileno(), stride, self.start + begin + int(row) * stride) for row in rows)
            shape = [len(rows), *shape[1:]]
        array = np.frombuffer(data, "u1" if quantized else DTYPES[spec["dtype"]]).reshape(shape)
        value = mx.array(array)
        if spec["dtype"] == "BF16":
            value = value.view(mx.bfloat16)
        if self.resident:
            self.cache[cache_key] = value
        return value

    def __getitem__(self, key):
        return self.read(key)

    def __contains__(self, key):
        return key in self.header

    def embedding(self, key, ids):
        value = self.read(key, ids)
        if value.dtype == mx.int8:
            scale = self.read(key + "_scale", ids)
            value = rotate(value.astype(mx.float32) * scale).astype(mx.bfloat16)
        return value

    def raw_linear(self, prefix, x):
        key = prefix + ".weight"
        filename = self.raw_index[key]
        if filename not in self.raw_files:
            directory = self.raw_source / self.raw_component
            self.raw_files[filename] = SafeTensorFile(directory / filename)
        parsed = self.raw_files[filename]
        info = parsed.tensors[key]
        array = np.asarray(parsed.memmap(info))
        weight = mx.array(array)
        if info.dtype == "BF16":
            weight = weight.view(mx.bfloat16)
        y = x @ weight.astype(x.dtype).T
        if prefix + ".bias" in self.header:
            y = y + self[prefix + ".bias"].astype(x.dtype)
        return y

    def q4_linear(self, prefix, x):
        spec = self.header[prefix + ".weight"]
        cache_key = (prefix, "mlx-q4", str(x.dtype))
        if cache_key in self.cache:
            packed, scale, bias = self.cache[cache_key]
        else:
            packed_np = np.asarray(self[prefix + ".weight"])
            rows, columns = packed_np.shape[0], packed_np.shape[1] * 2
            values = np.stack((packed_np & 15, packed_np >> 4), axis=2).reshape(rows, columns // 8, 8).astype(np.uint32)
            shifts = (np.arange(8, dtype=np.uint32) * 4).reshape(1, 1, 8)
            packed32 = np.bitwise_or.reduce(values << shifts, axis=2)
            self.cache.pop((prefix + ".weight", None), None)
            begin, end = spec["scale_offsets"]
            scale_np = np.frombuffer(os.pread(self.file.fileno(), end - begin, self.start + begin), "<f4").reshape(rows, -1)
            packed = mx.array(packed32)
            scale = mx.array(scale_np).astype(x.dtype)
            bias = mx.array(scale_np * -8).astype(x.dtype)
            if self.resident:
                self.cache[cache_key] = (packed, scale, bias)
        group_size = int(spec["dtype"].removeprefix("Q4S"))
        y = mx.quantized_matmul(x, packed, scale, bias, transpose=True, group_size=group_size, bits=4)
        if prefix + ".bias" in self.header:
            y = y + self[prefix + ".bias"].astype(x.dtype)
        return y

    def linear(self, prefix, x):
        if self.header[prefix + ".weight"]["dtype"].startswith("Q4S"):
            return self.raw_linear(prefix, x) if self.raw_source else self.q4_linear(prefix, x)
        weight = self[prefix + ".weight"]
        if weight.dtype == mx.int8:
            scale = self[prefix + ".weight_scale"]
            rotated = rotate(x)
            restored = weight.astype(mx.float32) * scale
            y = rotated.astype(x.dtype) @ restored.astype(x.dtype).T
        else:
            y = x @ weight.astype(x.dtype).T
        if prefix + ".bias" in self.header:
            y = y + self[prefix + ".bias"].astype(x.dtype)
        return y


def norm(x, weight=None, eps=1e-5):
    return mx.fast.rms_norm(x, None if weight is None else weight.astype(x.dtype), eps)


def heads(x, head_dim=128):
    return x.reshape(1, x.shape[0], -1, head_dim).transpose(0, 2, 1, 3)


def attention(q, k, v, mask=None, scale=None):
    output = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale or q.shape[-1] ** -0.5, mask=mask)
    return output.transpose(0, 2, 1, 3).reshape(q.shape[2], -1)


def rope(x, angles):
    count = angles.shape[-1]
    cosine = mx.cos(angles).astype(x.dtype)
    sine = mx.sin(angles).astype(x.dtype)
    first, second = x[..., :count], x[..., count:2 * count]
    return mx.concatenate([first * cosine - second * sine, first * sine + second * cosine, x[..., 2 * count:]], axis=-1)


def swiglu(x):
    gate, up = mx.split(x, 2, axis=-1)
    return gate * mx.sigmoid(gate) * up
