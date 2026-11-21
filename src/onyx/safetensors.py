import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DTYPE_INFO = {
    "BOOL": (1, np.dtype("?")),
    "I8": (1, np.dtype("i1")),
    "U8": (1, np.dtype("u1")),
    "I16": (2, np.dtype("<i2")),
    "U16": (2, np.dtype("<u2")),
    "F16": (2, np.dtype("<f2")),
    "BF16": (2, np.dtype("<u2")),
    "I32": (4, np.dtype("<i4")),
    "U32": (4, np.dtype("<u4")),
    "F32": (4, np.dtype("<f4")),
    "I64": (8, np.dtype("<i8")),
    "U64": (8, np.dtype("<u8")),
    "F64": (8, np.dtype("<f8")),
}


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dtype: str
    shape: tuple[int, ...]
    begin: int
    end: int
    data_begin: int

    @property
    def element_count(self):
        return math.prod(self.shape)

    @property
    def byte_size(self):
        return self.end - self.begin

    @property
    def absolute_begin(self):
        return self.data_begin + self.begin

    @property
    def absolute_end(self):
        return self.data_begin + self.end


class SafeTensorFile:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.size = self.path.stat().st_size
        with self.path.open("rb") as stream:
            raw_size = stream.read(8)
            if len(raw_size) != 8:
                raise ValueError(f"Truncated SafeTensors header: {path}")
            self.header_size = struct.unpack("<Q", raw_size)[0]
            if self.header_size > min(self.size - 8, 256 * 1024 * 1024):
                raise ValueError(f"Invalid SafeTensors header size: {path}")
            self.header = json.loads(stream.read(self.header_size))
        self.data_begin = 8 + self.header_size
        self.tensors = {}
        for name, spec in self.header.items():
            if name == "__metadata__":
                continue
            if spec["dtype"] not in DTYPE_INFO:
                raise ValueError(f"Unsupported dtype {spec['dtype']} for {name}")
            begin, end = spec["data_offsets"]
            expected = math.prod(spec["shape"]) * DTYPE_INFO[spec["dtype"]][0]
            if begin < 0 or end < begin or end - begin != expected or self.data_begin + end > self.size:
                raise ValueError(f"Invalid tensor range: {name}")
            self.tensors[name] = TensorInfo(name, spec["dtype"], tuple(spec["shape"]), begin, end, self.data_begin)
        ranges = sorted((info.begin, info.end, name) for name, info in self.tensors.items())
        for previous, current in zip(ranges, ranges[1:]):
            if previous[1] > current[0]:
                raise ValueError(f"Overlapping tensors: {previous[2]} and {current[2]}")

    def memmap(self, info: TensorInfo):
        _, dtype = DTYPE_INFO[info.dtype]
        return np.memmap(
            self.path,
            dtype=dtype,
            mode="r",
            offset=info.absolute_begin,
            shape=info.shape,
            order="C",
        )

    def read_raw(self, info: TensorInfo) -> bytes:
        with self.path.open("rb") as stream:
            stream.seek(info.absolute_begin)
            data = stream.read(info.byte_size)
        if len(data) != info.byte_size:
            raise OSError(f"Truncated tensor: {info.name}")
        return data


def component_index(source: Path, component: str, required_names=None):
    directory = source / component
    index = json.loads((directory / "model.safetensors.index.json").read_text())
    filenames = None
    if required_names:
        filenames = set()
        for name in required_names:
            if name not in index["weight_map"]:
                raise KeyError(f"Unknown tensor: {name}")
            filenames.add(index["weight_map"][name])
    files = {}
    for name, filename in index["weight_map"].items():
        if filenames is not None and filename not in filenames:
            continue
        path = directory / filename
        if path not in files:
            files[path] = SafeTensorFile(path)
    return index, files
