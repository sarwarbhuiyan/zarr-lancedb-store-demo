"""Zarr stores that read and write Lance tables through a LanceDB Enterprise server.

The row layouts are the same as `LanceStore` (one row per Zarr key) and `LanceTensorStore`
(one chip per row), but every read is a REST take by row ID. The query node routes each take
to the plan executor that owns the data, so repeated reads come from the executor's page cache
instead of object storage.

Each store pins one table version when it opens, the given one or the latest, the way a Zarr
reader expects a fixed snapshot. Pinning also matters for speed: under the server's default strong
consistency, a request against the latest version first checks object storage for a newer one,
which costs a storage round trip on every read; a request for a pinned version skips it.

Two REST behaviours shape the code: a take returns rows sorted by row ID rather than in request
order, so results are matched back by `_rowid`; and a take on a Blob V2 column returns the blob
descriptors, not the bytes. Blob tables are therefore read through remote `BlobFile` handles, one
REST request per byte range, served by the query node without the plan executor's cache (ENT-3100).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterable
from concurrent.futures import ThreadPoolExecutor

import lancedb
import pyarrow as pa
from zarr.abc.store import ByteRequest, OffsetByteRequest, RangeByteRequest, Store, SuffixByteRequest
from zarr.core.buffer import Buffer, BufferPrototype

from lance_zarr.batching import TickBatcher
from lance_zarr.store import _blob_range, _slice
from lance_zarr.tensor_store import LanceTensorStore

DEFAULT_REGION = "us-east-1"


def connect_table(table: str, *, database: str, host: str, api_key: str, region: str = DEFAULT_REGION,
                  version: int | str | None = None):
    """Open a remote table pinned to `version` (a number or tag), or to the latest version now."""
    db = lancedb.connect(f"db://{database}", host_override=host, api_key=api_key, region=region)
    tbl = db.open_table(table)
    tbl.checkout(tbl.version if version is None else version)
    return tbl


def scan_with_row_ids(tbl, columns: list[str]) -> pa.Table:
    """Every row of `columns` plus `_rowid`. A remote query needs an explicit limit to return all rows."""
    return tbl.search().select(columns).with_row_id(True).limit(tbl.count_rows()).to_arrow()


def take_in_order(tbl, row_ids: list[int], column: str) -> pa.Array:
    """`column` for `row_ids`, in the order given (the REST take answers in row-ID order)."""
    got = tbl.take_row_ids(row_ids).select([column]).with_row_id().to_arrow()
    position = {rid: i for i, rid in enumerate(got.column("_rowid").to_pylist())}
    return got.column(column).take(pa.array([position[r] for r in row_ids])).combine_chunks()


_DELETED = object()
_SCHEMA = pa.schema([pa.field("key", pa.string(), nullable=False), pa.field("value", pa.large_binary())])
# MemWAL needs an unenforced primary key. The SDK will not set one on a remote table, so a WAL
# table declares it in the field metadata Lance reads when the table is created.
_LSM_SCHEMA = pa.schema([
    pa.field("key", pa.string(), nullable=False,
             metadata={"lance-schema:unenforced-primary-key": "true",
                       "lance-schema:unenforced-primary-key:position": "0"}),
    pa.field("value", pa.large_binary()),
])
# A WAL-backed table cannot report a base-plus-memtable row count, so key scans use a fixed cap.
_LSM_SCAN_LIMIT = 100_000_000
# The WAL shard rejects a request body over 16 MiB (sophon_core::wal::DEFAULT_MAX_BODY_BYTES),
# and the query node reports that as the shard being unreachable, so LSM merges are split.
_LSM_MAX_REQUEST_BYTES = 8 * 1024 * 1024


def _sql_in(keys: Iterable[str]) -> str:
    return ", ".join("'" + k.replace("'", "''") + "'" for k in keys)


class LanceDBStore(Store):
    """A Zarr store over a `key`/`value` table (the `LanceStore` layout) served by LanceDB Enterprise.

    Read-only stores pin a version and read with REST takes by row ID. Writable stores buffer
    `set`/`delete` and `commit()` them as one REST `merge_insert` on `key` plus one delete. With
    `use_lsm=True` the merge goes through Enterprise's MemWAL: the table gets an unenforced primary
    key on `key` and an LSM write spec, writes land in the WAL shard instead of creating a Lance
    version each, and reads become key-filter queries, because a take by row ID cannot see rows
    still in the memtables.
    """

    supports_writes = True
    supports_deletes = True
    supports_listing = True
    supports_partial_writes = False

    def __init__(self, table: str, *, database: str, host: str, api_key: str, read_only: bool = True,
                 version: int | str | None = None, use_lsm: bool = False, flush_bytes: int = 256 * 1024 * 1024,
                 region: str = DEFAULT_REGION, blob_read_threads: int = 16) -> None:
        super().__init__(read_only=read_only or version is not None)
        self.table_name, self.database, self.host, self.api_key, self.region = table, database, host, api_key, region
        self.use_lsm, self.flush_bytes, self.requested_version = use_lsm, flush_bytes, version
        db = lancedb.connect(f"db://{database}", host_override=host, api_key=api_key, region=region)
        if table not in db.list_tables().tables:
            if self.read_only:
                raise FileNotFoundError(f"table {database}/{table} does not exist")
            db.create_table(table, schema=_LSM_SCHEMA if use_lsm else _SCHEMA)
        self._tbl = db.open_table(table)
        self._blob = "value" in self._tbl.blob_columns()
        if self._blob and (use_lsm or not self.read_only):
            raise NotImplementedError("Blob V2 value columns are read-only through this store")
        self.blob_read_threads = blob_read_threads
        self._blob_pool = ThreadPoolExecutor(max_workers=blob_read_threads)
        if use_lsm and self._tbl.get_lsm_write_spec() is None:
            self._tbl.set_lsm_write_spec(lancedb.LsmWriteSpec.unsharded().with_maintained_indexes([]))
        self._buffer: dict[str, bytes | object] = {}
        self._buffer_bytes = 0
        self.rest_reads = 0
        self.commits = 0
        self._load()
        self._batcher: TickBatcher[tuple[str, ByteRequest | None], bytes | None] = TickBatcher(self._fetch)

    def _load(self) -> None:
        """Rebuild the key map. LSM tables read the latest state (memtables have no version)."""
        if self.use_lsm:
            keys = self._tbl.search().select(["key"]).limit(_LSM_SCAN_LIMIT).to_arrow()
            self._index: dict[str, int | None] = dict.fromkeys(keys.column("key").to_pylist())
        else:
            if self.requested_version is None:
                self._tbl.checkout_latest()
            self._tbl.checkout(self._tbl.version if self.requested_version is None else self.requested_version)
            keys = scan_with_row_ids(self._tbl, ["key"])
            self._index = dict(zip(keys.column("key").to_pylist(), keys.column("_rowid").to_pylist()))
            if self._blob:
                # One handle per row, opened once; each knows its blob's size for suffix ranges.
                files = self._tbl.fetch_blob_files("value", list(self._index.values()))
                self._blob_files = dict(zip(self._index, files))
                self._blob_sizes = {k: 0 if f is None else f.size() for k, f in self._blob_files.items()}
        self.version = None if self.use_lsm else self._tbl.version

    # ---- reads ---------------------------------------------------------------------

    def _read_blob(self, request: tuple[str, ByteRequest | None]) -> bytes | None:
        key, byte_range = request
        f = self._blob_files.get(key)
        if f is None:
            return None
        start, length = _blob_range(byte_range, self._blob_sizes[key])
        return b"" if length <= 0 else bytes(f.read_range(start, length))

    def _fetch(self, requests: list[tuple[str, ByteRequest | None]]) -> list[bytes | None]:
        self.rest_reads += 1
        if self._blob:
            return list(self._blob_pool.map(self._read_blob, requests))
        keys = list(dict.fromkeys(k for k, _ in requests))
        if self.use_lsm:
            got = (self._tbl.search().where(f"key IN ({_sql_in(keys)})").select(["key", "value"])
                   .limit(len(keys)).to_arrow())
            values = dict(zip(got.column("key").to_pylist(), got.column("value").to_pylist()))
        else:
            values = dict(zip(keys, take_in_order(self._tbl, [self._index[k] for k in keys], "value").to_pylist()))
        return [None if values.get(k) is None else _slice(values[k], r) for k, r in requests]

    async def get(self, key: str, prototype: BufferPrototype, byte_range: ByteRequest | None = None) -> Buffer | None:
        if byte_range is not None and not isinstance(
            byte_range, (RangeByteRequest, OffsetByteRequest, SuffixByteRequest)
        ):
            raise TypeError(f"Unexpected byte_range, got {byte_range!r}")
        if key in self._buffer:
            v = self._buffer[key]
            return None if v is _DELETED else prototype.buffer.from_bytes(_slice(v, byte_range))
        if key not in self._index:
            return None
        data = await self._batcher.submit((key, byte_range))
        return None if data is None else prototype.buffer.from_bytes(data)

    async def get_partial_values(self, prototype: BufferPrototype,
                                 key_ranges: Iterable[tuple[str, ByteRequest | None]]) -> list[Buffer | None]:
        return list(await asyncio.gather(*(self.get(k, prototype, r) for k, r in key_ranges)))

    async def exists(self, key: str) -> bool:
        if key in self._buffer:
            return self._buffer[key] is not _DELETED
        return key in self._index

    # ---- writes --------------------------------------------------------------------

    def _check_writable_sync(self) -> None:
        if self.read_only:
            raise ValueError("store is read-only")

    async def set(self, key: str, value: Buffer) -> None:
        self._check_writable_sync()
        data = value.to_bytes()
        self._buffer[key] = data
        self._buffer_bytes += len(data)
        if self._buffer_bytes >= self.flush_bytes:
            await asyncio.get_running_loop().run_in_executor(None, self.commit)

    async def delete(self, key: str) -> None:
        self._check_writable_sync()
        self._buffer[key] = _DELETED

    def commit(self) -> None:
        """Write buffered sets as one REST merge_insert on `key`, then buffered deletes as one delete."""
        self._check_writable_sync()
        buffer, self._buffer, self._buffer_bytes = self._buffer, {}, 0
        upserts = {k: v for k, v in buffer.items() if v is not _DELETED}
        deletes = [k for k, v in buffer.items() if v is _DELETED]
        if not self.use_lsm and (upserts or deletes):
            self._tbl.checkout_latest()  # a version-pinned table reference rejects writes
        for part in self._request_parts(upserts):
            batch = pa.table({"key": list(part), "value": list(part.values())},
                             schema=_LSM_SCHEMA if self.use_lsm else _SCHEMA)
            merge = self._tbl.merge_insert("key").when_matched_update_all().when_not_matched_insert_all()
            if self.use_lsm:
                merge = merge.use_lsm(True)
            merge.execute(batch)
        if deletes:
            self._tbl.delete(f"key IN ({_sql_in(deletes)})")
        if upserts or deletes:
            self.commits += 1
            self._load()

    def _request_parts(self, upserts: dict[str, bytes]) -> list[dict[str, bytes]]:
        """One part per merge_insert request: all of it, or slices under the WAL body cap."""
        if not upserts or not self.use_lsm:
            return [upserts] if upserts else []
        parts, part, size = [], {}, 0
        for k, v in upserts.items():
            if part and size + len(v) > _LSM_MAX_REQUEST_BYTES:
                parts.append(part)
                part, size = {}, 0
            part[k] = v
            size += len(v)
        parts.append(part)
        return parts

    def close(self) -> None:
        if not self.read_only and self._buffer:
            self.commit()
        super().close()

    # ---- listing -------------------------------------------------------------------

    def _all_keys(self) -> list[str]:
        keys = dict.fromkeys(self._index)
        for k, v in self._buffer.items():
            if v is _DELETED:
                keys.pop(k, None)
            else:
                keys[k] = None
        return list(keys)

    async def list(self) -> AsyncIterator[str]:
        for k in self._all_keys():
            yield k

    async def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        for k in self._all_keys():
            if k.startswith(prefix):
                yield k

    async def list_dir(self, prefix: str) -> AsyncIterator[str]:
        base = prefix.rstrip("/") + "/" if prefix.rstrip("/") else ""
        seen: set[str] = set()
        for k in self._all_keys():
            if k.startswith(base):
                child = k[len(base):].split("/", 1)[0]
                if child not in seen:
                    seen.add(child)
                    yield child

    def with_read_only(self, read_only: bool = False) -> LanceDBStore:
        return LanceDBStore(self.table_name, database=self.database, host=self.host, api_key=self.api_key,
                            read_only=read_only, version=self.requested_version, use_lsm=self.use_lsm,
                            flush_bytes=self.flush_bytes, region=self.region,
                            blob_read_threads=self.blob_read_threads)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, LanceDBStore) and (other.host, other.database, other.table_name, other.version) == (
            self.host, self.database, self.table_name, self.version)

    def __repr__(self) -> str:
        at = "lsm" if self.use_lsm else self.version
        return f"LanceDBStore({self.host}/{self.database}/{self.table_name}@{at})"


class LanceDBTensorStore(LanceTensorStore):
    """`LanceTensorStore` over a chip table, read through the REST API."""

    def __init__(self, table: str, *, database: str, host: str, api_key: str, version: int | str | None = None,
                 array: str = "reflectance", pixels_column: str = "pixels", attributes: dict | None = None,
                 region: str = DEFAULT_REGION) -> None:
        Store.__init__(self, read_only=True)
        self._tbl = connect_table(table, database=database, host=host, api_key=api_key, region=region,
                                  version=version)
        self.uri, self.version, self.storage_options = f"{host}/{database}/{table}", self._tbl.version, None
        keys = scan_with_row_ids(self._tbl, ["t", "ty", "tx"])
        self._setup(
            keys, keys.column("_rowid").to_pylist(), self._tbl.schema.field(pixels_column).type,
            array=array, pixels_column=pixels_column,
            attributes={**(attributes or {}), "lancedb_table": self.uri, "lancedb_version": self._tbl.version},
        )

    def _take(self, rows: list) -> pa.Array:
        return take_in_order(self._tbl, rows, self.pixels_column)
