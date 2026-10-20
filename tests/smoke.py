"""Round-trip a small array through LanceStore (binary and blob encodings, plain and sharded)."""

import shutil, tempfile, numpy as np, zarr
from lance_zarr import LanceStore

def roundtrip(encoding: str, shards):
    d = tempfile.mkdtemp()
    store = LanceStore(f"{d}/z.lance", value_encoding=encoding)
    data = np.arange(256 * 256, dtype="uint16").reshape(256, 256)
    arr = zarr.create_array(store, name="a", shape=data.shape, chunks=(32, 32), shards=shards, dtype="uint16")
    arr[:] = data
    v1 = store.commit()
    store.tag("first")
    arr[0:10, 0:10] = 7
    v2 = store.commit()
    ro = LanceStore(f"{d}/z.lance", version="first")
    old = zarr.open_array(ro, path="a", mode="r")
    new = zarr.open_array(LanceStore(f"{d}/z.lance", read_only=True), path="a", mode="r")
    assert np.array_equal(old[:], data), "tagged version should be unchanged"
    assert (new[0:10, 0:10] == 7).all() and np.array_equal(new[10:, 10:], data[10:, 10:])
    print(f"{encoding:6} shards={shards}: ok  versions {v1}->{v2}  lance reads for full read: {ro.stats.lance_reads}, keys {ro.stats.keys_read}")
    shutil.rmtree(d)

for enc in ("binary", "blob"):
    for shards in (None, (128, 128)):
        roundtrip(enc, shards)
