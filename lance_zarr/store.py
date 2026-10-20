"""A zarr-python v3 Store that keeps every Zarr key as one row of a Lance dataset.

Layout: one row per Zarr key (`zarr.json` metadata documents and encoded chunks alike).
Columns: `key` (string) and `value` (the encoded bytes, as a plain binary column or a
Lance Blob V2 column).

What it maps onto Lance:
- reads: concurrent `get` calls issued by zarr in the same event-loop tick are batched
  into one Lance read. With `key_lookup="map"` (the default) keys resolve through a key -> row
  map built when the store opens, the way Icechunk loads its manifest. With
  `key_lookup="index"` nothing is read at open: each batch is one `key IN (...)` filter
  answered by a BTREE on `key` (`create_key_index`), so the cost of a read does not grow
  with the number of keys in the table.
- byte ranges: with a Blob V2 value column, every read is a blob byte range. A `BlobFile`
  handle per row is opened at load from the blob descriptors, so a ranged read (how Zarr
  sharding reads a shard index and then its chunks) is one object-store request.
- writes: `set` and `delete` are buffered in memory and committed as one Lance version
  (merge_insert on `key`, then one delete) by `commit()` or `close()`.
- versions: a store opened with `version=` (number or tag) is a read-only snapshot.
"""

from __future__ import annotations

import asyncio
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
from typing import Literal

import lance
import pyarrow as pa
from zarr.abc.store import (
    ByteRequest,
    OffsetByteRequest,
    RangeByteRequest,
    Store,
    SuffixByteRequest,
)
from zarr.core.buffer import Buffer, BufferPrototype

from lance_zarr.batching import TickBatcher

ValueEncoding = Literal["binary", "blob"]
KeyLookup = Literal["map", "index"]
_DELETED = object()


def _schema(value_encoding: ValueEncoding) -> pa.Schema:
    if value_encoding == "blob":
        return pa.schema([pa.field("key", pa.string(), nullable=False), lance.blob_field("value")])
    return pa.schema(
        [pa.field("key", pa.string(), nullable=False), pa.field("value", pa.large_binary())]
    )


def _slice(data: bytes, byte_range: ByteRequest | None) -> bytes:
    if byte_range is None:
        return data
    if isinstance(byte_range, RangeByteRequest):
        return data[byte_range.start : byte_range.end]
    if isinstance(byte_range, OffsetByteRequest):
        return data[byte_range.offset :]
    if isinstance(byte_range, SuffixByteRequest):
        return data[-byte_range.suffix :] if byte_range.suffix else b""
    raise TypeError(f"Unexpected byte_range, got {byte_range!r}")


@dataclass
class _Stats:
    lance_reads: int = 0
    keys_read: int = 0
    commits: int = 0


def _blob_range(byte_range: ByteRequest | None, size: int) -> tuple[int, int]:
    """(start, length) inside a blob of `size` bytes for a zarr byte request."""
    if byte_range is None:
        return 0, size
    if isinstance(byte_range, RangeByteRequest):
        end = min(byte_range.end, size)
        return byte_range.start, end - byte_range.start
    if isinstance(byte_range, OffsetByteRequest):
        return byte_range.offset, size - byte_range.offset
    n = min(byte_range.suffix, size)
    return size - n, n


class LanceStore(Store):
    supports_writes = True
    supports_deletes = True
    supports_listing = True
    supports_partial_writes = False

    def __init__(
        self,
        uri: str,
        *,
        read_only: bool = False,
        version: int | str | None = None,
        value_encoding: ValueEncoding = "binary",
        storage_options: dict[str, str] | None = None,
        flush_bytes: int = 256 * 1024 * 1024,
        autocommit: bool = False,
        blob_read_threads: int = 16,
        key_lookup: KeyLookup = "map",
    ) -> None:
        super().__init__(read_only=read_only or version is not None)
        self.uri = uri
        self.version = version
        self.value_encoding = value_encoding
        self.storage_options = storage_options
        self.flush_bytes = flush_bytes
        self.autocommit = autocommit
        self.blob_read_threads = blob_read_threads
        self.key_lookup = key_lookup
        self._blob_pool = ThreadPoolExecutor(max_workers=blob_read_threads)
        self._blob_files: dict[int, lance.BlobFile | None] = {}
        self.stats = _Stats()
        self._buffer: dict[str, bytes | object] = {}
        self._inflight: dict[str, bytes | object] = {}
        self._buffer_bytes = 0
        self._commit_lock = threading.Lock()
        self._batcher: TickBatcher[tuple[str, ByteRequest | None], bytes | None] = TickBatcher(self._fetch_batch)
        self._ds: lance.LanceDataset | None = None
        self._index: dict[str, int] = {}
        self._load()

    # ---- dataset state -------------------------------------------------------------

    def _open_dataset(self) -> lance.LanceDataset | None:
        """The dataset at `version`, or None when nothing has been written yet."""
        if "://" not in self.uri:
            if not os.path.isdir(self.uri):
                return None
            return lance.dataset(self.uri, version=self.version, storage_options=self.storage_options)
        try:
            ds = lance.dataset(self.uri, storage_options=self.storage_options)
        except (ValueError, OSError, FileNotFoundError):
            return None
        return ds if self.version is None else ds.checkout_version(self.version)

    def _load(self) -> None:
        self._ds = self._open_dataset()
        self._index, self._blob_sizes, self._blob_files = {}, [], {}
        if self._ds is None:
            return
        if "value" in self._ds.schema.names:
            vtype = self._ds.schema.field("value").type
            is_blob = isinstance(vtype, pa.ExtensionType) and "blob" in vtype.extension_name
            self.value_encoding = "blob" if is_blob else "binary"
        if self.key_lookup == "index":
            return
        table = self._ds.to_table(columns=["key"])
        if self.value_encoding == "blob" and table.num_rows:
            # One BlobFile per row, opened from the blob descriptors in one read. Each handle
            # knows its blob's location and size, so every later byte range is one request.
            files = self._ds.take_blobs("value", indices=list(range(table.num_rows)))
            self._blob_files = dict(enumerate(files))
            self._blob_sizes = [0 if f is None else f.size() for f in files]
        self._index = {k: i for i, k in enumerate(table.column("key").to_pylist())}

    @property
    def dataset_version(self) -> int | None:
        return None if self._ds is None else self._ds.version

    def tag(self, name: str) -> None:
        if self._ds is None:
            raise ValueError("nothing committed yet")
        self._ds.tags.create(name, self._ds.version)

    # ---- writes --------------------------------------------------------------------

    def _check_writable_sync(self) -> None:
        if self.read_only:
            raise ValueError("store is read-only")

    def commit(self) -> int | None:
        """Write buffered sets and deletes as one Lance version; returns the version."""
        self._check_writable_sync()
        with self._commit_lock:
            if not self._buffer:
                return self.dataset_version
            batch, self._buffer, self._buffer_bytes = self._buffer, {}, 0
            self._inflight = batch
            try:
                self._write_batch(batch)
            finally:
                self._load()
                self._inflight = {}
            self.stats.commits += 1
            return self.dataset_version

    def _write_batch(self, batch: dict[str, bytes | object]) -> None:
        upserts = {k: v for k, v in batch.items() if v is not _DELETED}
        deletes = [k for k, v in batch.items() if v is _DELETED]
        if upserts:
            values: list = list(upserts.values())
            value_array = (
                lance.blob_array(values) if self.value_encoding == "blob" else pa.array(values, pa.large_binary())
            )
            table = pa.table(
                {"key": pa.array(list(upserts), pa.string()), "value": value_array},
                schema=_schema(self.value_encoding),
            )
            if self._ds is None:
                self._ds = lance.write_dataset(
                    table, self.uri, schema=_schema(self.value_encoding),
                    storage_options=self.storage_options, data_storage_version="2.2",
                )
            else:
                (
                    self._ds.merge_insert("key")
                    .when_matched_update_all()
                    .when_not_matched_insert_all()
                    .execute(table)
                )
        if deletes and self._ds is not None:
            quoted = ",".join("'" + k.replace("'", "''") + "'" for k in deletes)
            self._ds.delete(f"key IN ({quoted})")

    def create_key_index(self) -> None:
        """A BTREE on `key`; `key_lookup="index"` reads through it."""
        if self._ds is not None:
            self._ds.create_scalar_index("key", "BTREE", replace=True)

    async def set(self, key: str, value: Buffer) -> None:
        self._check_writable()
        data = value.to_bytes()
        self._buffer[key] = data
        self._buffer_bytes += len(data)
        if self.autocommit or self._buffer_bytes >= self.flush_bytes:
            await asyncio.to_thread(self.commit)

    async def set_if_not_exists(self, key: str, value: Buffer) -> None:
        if not await self.exists(key):
            await self.set(key, value)

    async def delete(self, key: str) -> None:
        self._check_writable()
        self._buffer[key] = _DELETED
        if self.autocommit:
            await asyncio.to_thread(self.commit)

    def close(self) -> None:
        if not self.read_only and self._buffer:
            self.commit()
        super().close()

    # ---- reads ---------------------------------------------------------------------

    def _fetch_batch(self, requests: list[tuple[str, ByteRequest | None]]) -> list[bytes | None]:
        """Resolve a batch of keys: one Lance take, or blob byte ranges from cached handles."""
        out: list[bytes | None] = [None] * len(requests)
        if self.key_lookup == "index":
            self._fetch_by_index(requests, out)
            return out
        rows = [(i, self._index.get(k), r) for i, (k, r) in enumerate(requests)]
        present = [(i, row, r) for i, row, r in rows if row is not None]
        if not present or self._ds is None:
            return out
        self.stats.lance_reads += 1
        self.stats.keys_read += len(present)
        if self.value_encoding == "blob":
            self._read_blob_ranges(present, out)
            return out
        table = self._ds.take([row for _, row, _ in present], columns=["value"])
        for (i, _, r), v in zip(present, table.column("value").to_pylist()):
            out[i] = None if v is None else _slice(v, r)
        return out

    @staticmethod
    def _key_filter(keys: Iterable[str]) -> str:
        return "key IN (" + ",".join("'" + k.replace("'", "''") + "'" for k in keys) + ")"

    def _fetch_by_index(self, requests: list[tuple[str, ByteRequest | None]], out: list) -> None:
        """One BTREE-filtered read for the batch; blob values then read by byte range."""
        if self._ds is None:
            return
        keys = list(dict.fromkeys(k for k, _ in requests))
        self.stats.lance_reads += 1
        if self.value_encoding == "binary":
            table = self._ds.to_table(columns=["key", "value"], filter=self._key_filter(keys))
            found = dict(zip(table.column("key").to_pylist(), table.column("value").to_pylist()))
            self.stats.keys_read += len(found)
            for i, (k, r) in enumerate(requests):
                v = found.get(k)
                out[i] = None if v is None else _slice(v, r)
            return
        table = self._ds.to_table(columns=["key"], filter=self._key_filter(keys), with_row_id=True)
        rows = dict(zip(table.column("key").to_pylist(), table.column("_rowid").to_pylist()))
        self.stats.keys_read += len(rows)
        ids = list(rows.values())
        files = dict(zip(ids, self._ds.take_blobs("value", ids=ids))) if ids else {}

        def read(item: tuple[int, tuple[str, ByteRequest | None]]) -> tuple[int, bytes | None]:
            i, (k, r) = item
            f = files.get(rows.get(k))
            if f is None:
                return i, None
            start, length = _blob_range(r, f.size())
            return i, b"" if length <= 0 else bytes(f.read_ranges([(start, length)])[0])

        for i, b in self._blob_pool.map(read, enumerate(requests)):
            out[i] = b

    def _read_blob_ranges(self, present: list[tuple[int, int, ByteRequest | None]], out: list) -> None:
        # Dataset.read_blob_ranges re-reads the blob descriptors on every call, a second
        # sequential request; the handles opened at load already hold them.
        by_row: dict[int, list[tuple[int, int, int]]] = {}
        for i, row, r in present:
            start, length = _blob_range(r, self._blob_sizes[row])
            if length <= 0:
                out[i] = b""
            else:
                by_row.setdefault(row, []).append((i, start, length))

        def read_row(item: tuple[int, list[tuple[int, int, int]]]) -> list[tuple[int, bytes | None]]:
            row, reqs = item
            f = self._blob_files[row]
            if f is None:
                return [(i, None) for i, _, _ in reqs]
            return [(i, bytes(b)) for (i, _, _), b in zip(reqs, f.read_ranges([(s, n) for _, s, n in reqs]))]

        for results in self._blob_pool.map(read_row, by_row.items()):
            for i, b in results:
                out[i] = b

    async def get(
        self, key: str, prototype: BufferPrototype, byte_range: ByteRequest | None = None
    ) -> Buffer | None:
        if not self._is_open:
            await self._open()
        if byte_range is not None and not isinstance(
            byte_range, (RangeByteRequest, OffsetByteRequest, SuffixByteRequest)
        ):
            raise TypeError(f"Unexpected byte_range, got {byte_range!r}")
        for staged in (self._buffer, self._inflight):
            if key in staged:
                v = staged[key]
                return None if v is _DELETED else prototype.buffer.from_bytes(_slice(v, byte_range))
        if self.key_lookup == "map" and key not in self._index:
            return None
        data = await self._batcher.submit((key, byte_range))
        return None if data is None else prototype.buffer.from_bytes(data)

    async def get_partial_values(
        self, prototype: BufferPrototype, key_ranges: Iterable[tuple[str, ByteRequest | None]]
    ) -> list[Buffer | None]:
        return list(await asyncio.gather(*(self.get(k, prototype, r) for k, r in key_ranges)))

    async def exists(self, key: str) -> bool:
        for staged in (self._buffer, self._inflight):
            if key in staged:
                return staged[key] is not _DELETED
        if self.key_lookup == "index":
            if self._ds is None:
                return False
            return await asyncio.to_thread(self._ds.count_rows, self._key_filter([key])) > 0
        return key in self._index

    # ---- listing -------------------------------------------------------------------

    def _all_keys(self) -> list[str]:
        if self.key_lookup == "index" and self._ds is not None:
            keys = set(self._ds.to_table(columns=["key"]).column("key").to_pylist())
        else:
            keys = set(self._index)
        for k, v in [*self._inflight.items(), *self._buffer.items()]:
            if v is _DELETED:
                keys.discard(k)
            else:
                keys.add(k)
        return sorted(keys)

    async def list(self) -> AsyncIterator[str]:
        for k in self._all_keys():
            yield k

    async def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        for k in self._all_keys():
            if k.startswith(prefix):
                yield k

    async def list_dir(self, prefix: str) -> AsyncIterator[str]:
        prefix = prefix.rstrip("/")
        base = f"{prefix}/" if prefix else ""
        seen: set[str] = set()
        for k in self._all_keys():
            if k.startswith(base):
                child = k[len(base) :].split("/", 1)[0]
                if child and child not in seen:
                    seen.add(child)
                    yield child

    # ---- identity ------------------------------------------------------------------

    def with_read_only(self, read_only: bool = False) -> LanceStore:
        return LanceStore(
            self.uri, read_only=read_only, version=self.version,
            value_encoding=self.value_encoding, storage_options=self.storage_options,
            autocommit=self.autocommit, blob_read_threads=self.blob_read_threads,
            key_lookup=self.key_lookup,
        )

    def __getstate__(self) -> dict:
        return {
            "uri": self.uri, "read_only": self.read_only, "version": self.version,
            "value_encoding": self.value_encoding, "storage_options": self.storage_options,
            "flush_bytes": self.flush_bytes, "autocommit": self.autocommit,
            "blob_read_threads": self.blob_read_threads, "key_lookup": self.key_lookup,
        }

    def __setstate__(self, state: dict) -> None:
        self.__init__(state.pop("uri"), **state)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, LanceStore)
            and other.uri == self.uri
            and other.version == self.version
            and other.read_only == self.read_only
        )

    def __repr__(self) -> str:
        return f"LanceStore({self.uri!r}, version={self.version!r}, encoding={self.value_encoding})"
