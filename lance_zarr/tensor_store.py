"""A read-only Zarr view over a Lance table whose rows are decoded image chips.

The table holds one row per (time, tile row, tile column) chip, with the pixels in a
fixed-shape tensor column of shape (bands, chip, chip). The store synthesizes the Zarr
metadata for an array of shape (time, bands, height, width) chunked one chip per chunk
with the plain `bytes` codec, and serves each chunk key from the matching row. Zarr and
xarray assemble arbitrary windows across chips; the same rows stay directly usable from
Lance (numpy, SQL filters) and from LanceDB vector search on any embedding column.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator, Iterable

import lance
import numpy as np
import pyarrow as pa
from zarr.abc.store import ByteRequest, Store
from zarr.core.buffer import Buffer, BufferPrototype

from lance_zarr.batching import TickBatcher
from lance_zarr.store import _slice

_CHUNK = re.compile(r"^(?P<array>[^/]+)/c/(?P<t>\d+)/0/(?P<i>\d+)/(?P<j>\d+)$")


class LanceTensorStore(Store):
    supports_writes = False
    supports_deletes = False
    supports_listing = True
    supports_partial_writes = False

    def __init__(
        self,
        uri: str,
        *,
        array: str = "reflectance",
        pixels_column: str = "pixels",
        version: int | str | None = None,
        storage_options: dict[str, str] | None = None,
        attributes: dict | None = None,
    ) -> None:
        super().__init__(read_only=True)
        self.uri, self.version, self.storage_options = uri, version, storage_options
        self._ds = lance.dataset(uri, version=version, storage_options=storage_options)
        keys = self._ds.to_table(columns=["t", "ty", "tx"])
        self._setup(
            keys, list(range(keys.num_rows)), self._ds.schema.field(pixels_column).type,
            array=array, pixels_column=pixels_column,
            attributes={**(attributes or {}), "lance_uri": uri, "lance_version": self._ds.version},
        )

    def _setup(self, keys: pa.Table, rows: list, tensor_type: pa.DataType, *, array: str,
               pixels_column: str, attributes: dict) -> None:
        """Index chips by (t, ty, tx) and synthesize the Zarr metadata from the tensor type.

        `rows` holds, per key row, whatever `_take` uses to address it: a dataset index here,
        a row ID for a remote table.
        """
        self.array, self.pixels_column = array, pixels_column
        self.lance_reads = 0
        t, ty, tx = (keys.column(c).to_numpy() for c in ("t", "ty", "tx"))
        self._index = {(int(a), int(b), int(c)): r for r, a, b, c in zip(rows, t, ty, tx)}
        bands, ch, cw = tensor_type.shape
        self._dtype = np.dtype(tensor_type.value_type.to_pandas_dtype())
        nt, ny, nx = int(t.max()) + 1, int(ty.max()) + 1, int(tx.max()) + 1
        meta = {
            "zarr_format": 3,
            "node_type": "array",
            "shape": [nt, bands, ny * ch, nx * cw],
            "data_type": self._dtype.name,
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [1, bands, ch, cw]}},
            "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
            "fill_value": 0,
            "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
            "attributes": attributes,
            "dimension_names": ["time", "band", "y", "x"],
        }
        self._meta = {
            "zarr.json": json.dumps({"zarr_format": 3, "node_type": "group", "attributes": {}}).encode(),
            f"{array}/zarr.json": json.dumps(meta).encode(),
        }
        self._batcher: TickBatcher[tuple[tuple[int, int, int], ByteRequest | None], bytes] = TickBatcher(self._fetch)

    def _take(self, rows: list) -> pa.Array:
        """The pixels column for `rows`, in the order given."""
        return self._ds.take(rows, columns=[self.pixels_column]).column(self.pixels_column).combine_chunks()

    def _fetch(self, requests: list[tuple[tuple[int, int, int], ByteRequest | None]]) -> list[bytes]:
        self.lance_reads += 1
        nd = self._take([self._index[c] for c, _ in requests]).to_numpy_ndarray()
        le = self._dtype.newbyteorder("<")
        return [_slice(np.ascontiguousarray(nd[k], dtype=le).tobytes(), r) for k, (_, r) in enumerate(requests)]

    async def get(self, key: str, prototype: BufferPrototype, byte_range: ByteRequest | None = None) -> Buffer | None:
        if key in self._meta:
            return prototype.buffer.from_bytes(_slice(self._meta[key], byte_range))
        m = _CHUNK.match(key)
        if not m or m["array"] != self.array:
            return None
        coord = (int(m["t"]), int(m["i"]), int(m["j"]))
        if coord not in self._index:
            return None
        return prototype.buffer.from_bytes(await self._batcher.submit((coord, byte_range)))

    async def get_partial_values(self, prototype: BufferPrototype, key_ranges: Iterable[tuple[str, ByteRequest | None]]) -> list[Buffer | None]:
        return list(await asyncio.gather(*(self.get(k, prototype, r) for k, r in key_ranges)))

    async def exists(self, key: str) -> bool:
        if key in self._meta:
            return True
        m = _CHUNK.match(key)
        return bool(m) and (int(m["t"]), int(m["i"]), int(m["j"])) in self._index

    async def set(self, key: str, value: Buffer) -> None:
        raise ValueError("LanceTensorStore is read-only; write chips through Lance or LanceDB")

    async def delete(self, key: str) -> None:
        raise ValueError("LanceTensorStore is read-only")

    def _keys(self) -> list[str]:
        return list(self._meta) + [f"{self.array}/c/{t}/0/{i}/{j}" for (t, i, j) in self._index]

    async def list(self) -> AsyncIterator[str]:
        for k in self._keys():
            yield k

    async def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        for k in self._keys():
            if k.startswith(prefix):
                yield k

    async def list_dir(self, prefix: str) -> AsyncIterator[str]:
        base = prefix.rstrip("/") + "/" if prefix.rstrip("/") else ""
        seen: set[str] = set()
        for k in self._keys():
            if k.startswith(base):
                child = k[len(base):].split("/", 1)[0]
                if child not in seen:
                    seen.add(child)
                    yield child

    def __eq__(self, other: object) -> bool:
        return isinstance(other, LanceTensorStore) and (other.uri, other.version, other.array) == (self.uri, self.version, self.array)
