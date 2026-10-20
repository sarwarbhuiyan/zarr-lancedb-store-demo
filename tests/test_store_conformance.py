"""Run zarr-python's own Store conformance suite against LanceStore.

The suite assumes every `set` is immediately durable and visible to a second store
instance, so it runs with `autocommit=True` (one Lance version per write). The default
buffered mode commits once per `commit()`/`close()`, like an Icechunk session.
"""

import lance
import pyarrow as pa
import pytest
from zarr.core.buffer import Buffer, cpu
from zarr.testing.store import StoreTests

from lance_zarr import LanceStore
from lance_zarr.store import _schema


@pytest.mark.parametrize("key_lookup", ["map", "index"], indirect=False)
@pytest.mark.parametrize("encoding", ["binary", "blob"], indirect=False)
class TestLanceStore(StoreTests[LanceStore, cpu.Buffer]):
    store_cls = LanceStore
    buffer_cls = cpu.Buffer

    @pytest.fixture
    def store_kwargs(self, tmp_path, encoding, key_lookup):
        return {
            "uri": str(tmp_path / "store.lance"), "value_encoding": encoding, "autocommit": True,
            "key_lookup": key_lookup,
        }

    async def set(self, store: LanceStore, key: str, value: Buffer) -> None:
        values = [value.to_bytes()]
        col = lance.blob_array(values) if store.value_encoding == "blob" else pa.array(values, pa.large_binary())
        table = pa.table({"key": [key], "value": col}, schema=_schema(store.value_encoding))
        try:
            ds = lance.dataset(store.uri)
            ds.merge_insert("key").when_matched_update_all().when_not_matched_insert_all().execute(table)
        except (ValueError, OSError):
            lance.write_dataset(table, store.uri, schema=_schema(store.value_encoding), data_storage_version="2.2")
        store._load()

    async def get(self, store: LanceStore, key: str) -> Buffer:
        ds = lance.dataset(store.uri)
        rows = ds.to_table(columns=["key"]).column("key").to_pylist()
        idx = rows.index(key)
        if store.value_encoding == "blob":
            data = ds.read_blobs("value", indices=[idx])[0][1]
        else:
            data = ds.take([idx], columns=["value"]).column("value")[0].as_py()
        return self.buffer_cls.from_bytes(data)

    def test_store_repr(self, store: LanceStore) -> None:
        assert "LanceStore(" in repr(store)

    def test_store_supports_writes(self, store: LanceStore) -> None:
        assert store.supports_writes

    def test_store_supports_listing(self, store: LanceStore) -> None:
        assert store.supports_listing
