"""Shared pieces for the benchmark: synthetic data, layout builders, and target config."""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass

import icechunk
import lance
import numpy as np
import pyarrow as pa
import zarr
from zarr.codecs import BloscCodec

from lance_zarr import LanceStore

T, B, H, W = 12, 14, 1024, 1024
BIG = (1, B, 512, 512)
SMALL = (1, B, 64, 64)
CHIP = 64
EMB_DIM = 64
# Blosc + byte shuffle + zstd is the common choice for EO reflectance arrays
COMPRESSOR = BloscCodec(cname="zstd", clevel=5, shuffle="shuffle")
ROOT = os.environ.get("BENCH_ROOT", os.path.expanduser("~/dev/zarr-lancedb-store-demo/data"))

# MinIO direct (writes) and through toxiproxy (reads with injected latency)
S3_DIRECT = os.environ.get("S3_DIRECT", "http://localhost:19000")
S3_PROXY = os.environ.get("S3_PROXY", "http://localhost:19001")
S3_KEY, S3_SECRET, S3_BUCKET, S3_REGION = "minioadmin", "minioadmin", "bench", "us-east-1"
# LanceDB Enterprise serves tables at <root>/<database>/<table>.lance; with root s3://bench the
# Lance layouts live in database LANCE_DB so the server sees them as tables.
LANCE_DB = "zarr"


@dataclass(frozen=True)
class Target:
    name: str  # "local" or "s3"
    endpoint: str | None = None

    def lance_uri(self, layout: str) -> str:
        return f"{ROOT}/{layout}.lance" if self.name == "local" else f"s3://{S3_BUCKET}/{LANCE_DB}/{layout}.lance"

    def lance_opts(self) -> dict | None:
        if self.name == "local":
            return None
        return {
            "aws_access_key_id": S3_KEY, "aws_secret_access_key": S3_SECRET,
            "aws_endpoint": self.endpoint, "aws_region": S3_REGION, "allow_http": "true",
            "aws_virtual_hosted_style_request": "false",
        }

    def zarr_store(self, layout: str, read_only: bool):
        if self.name == "local":
            return zarr.storage.LocalStore(f"{ROOT}/{layout}.zarr", read_only=read_only)
        from obstore.store import S3Store

        s3 = S3Store(
            S3_BUCKET, prefix=f"{layout}.zarr", endpoint=self.endpoint, access_key_id=S3_KEY,
            secret_access_key=S3_SECRET, region=S3_REGION, virtual_hosted_style_request=False,
            client_options={"allow_http": True},
        )
        return zarr.storage.ObjectStore(s3, read_only=read_only)

    def icechunk_storage(self, layout: str):
        if self.name == "local":
            return icechunk.local_filesystem_storage(f"{ROOT}/{layout}.icechunk")
        return icechunk.s3_storage(
            bucket=S3_BUCKET, prefix=f"{layout}.icechunk", endpoint_url=self.endpoint,
            access_key_id=S3_KEY, secret_access_key=S3_SECRET, region=S3_REGION,
            allow_http=True, force_path_style=True,
        )


def synthetic_cube(seed: int = 0) -> np.ndarray:
    """Smooth reflectance-like fields (uint16, 0..10000) so compression behaves realistically."""
    rng = np.random.default_rng(seed)
    coarse = rng.uniform(500, 6000, size=(B, H // 32, W // 32)).astype(np.float32)
    base = np.kron(coarse, np.ones((32, 32), np.float32))
    k = np.ones(9, np.float32) / 9
    for axis in (1, 2):
        base = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), axis, base)
    cube = np.empty((T, B, H, W), np.uint16)
    for t in range(T):
        season = 1.0 + 0.25 * np.sin(2 * np.pi * t / T)
        noise = rng.normal(0, 40, size=(B, H, W)).astype(np.float32)
        cube[t] = np.clip(base * season + noise, 0, 10000).astype(np.uint16)
    return cube


def chip_rows(cube: np.ndarray, seed: int = 1) -> pa.Table:
    """One row per (date, tile row, tile col) chip: pixels as a tensor plus search columns."""
    rng = np.random.default_rng(seed)
    proj = rng.normal(size=(B, EMB_DIM)).astype(np.float32)
    ny, nx = H // CHIP, W // CHIP
    t_idx, ty_idx, tx_idx, chips, embs, cloud = [], [], [], [], [], []
    for t in range(T):
        for i in range(ny):
            for j in range(nx):
                c = cube[t, :, i * CHIP:(i + 1) * CHIP, j * CHIP:(j + 1) * CHIP]
                feat = c.reshape(B, -1).mean(axis=1) / 10000.0
                e = feat @ proj + rng.normal(0, 0.01, EMB_DIM)
                t_idx.append(t); ty_idx.append(i); tx_idx.append(j); chips.append(c)
                embs.append((e / np.linalg.norm(e)).astype(np.float32))
                cloud.append(float(rng.beta(1, 6)))
    pixels = pa.FixedShapeTensorArray.from_numpy_ndarray(np.stack(chips))
    dates = [f"2025-{1 + t:02d}-15" for t in t_idx]
    return pa.table({
        "t": pa.array(t_idx, pa.int32()), "ty": pa.array(ty_idx, pa.int32()), "tx": pa.array(tx_idx, pa.int32()),
        "acquired": pa.array(dates), "cloud_frac": pa.array(cloud, pa.float32()),
        "x0": pa.array([j * CHIP for j in tx_idx], pa.int32()), "y0": pa.array([i * CHIP for i in ty_idx], pa.int32()),
        "vector": pa.FixedSizeListArray.from_arrays(pa.array(np.concatenate(embs)), EMB_DIM),
        "pixels": pixels,
    })


def dir_size(path: str) -> int:
    total = 0
    for dp, _, fns in os.walk(path):
        total += sum(os.path.getsize(os.path.join(dp, f)) for f in fns)
    return total


def write_zarr(store, cube: np.ndarray, chunks, shards) -> None:
    arr = zarr.create_array(store, name="reflectance", shape=cube.shape, chunks=chunks, shards=shards,
                            dtype="uint16", compressors=COMPRESSOR, dimension_names=["time", "band", "y", "x"],
                            overwrite=True)
    arr[:] = cube


LAYOUTS = {
    "zarr_big": "Zarr, 512 px chunks, blosc-zstd",
    "zarr_sharded": "Zarr, 64 px chunks in 512 px shards, blosc-zstd",
    "icechunk_small": "Icechunk, 64 px chunks, blosc-zstd",
    "lance_big": "LanceStore, 512 px chunks, blosc-zstd, binary column",
    "lance_small": "LanceStore, 64 px chunks, blosc-zstd, binary column",
    "lance_sharded_blob": "LanceStore, 64 px chunks in 512 px shards, Blob V2 ranged reads",
    "lance_tensor": "LanceTensorStore over a chip table (64 px tensor column + vector + filters)",
    "zarr_small": "Zarr, 64 px chunks, blosc-zstd, one object per chunk",
    "lance_ref_small": "LanceStore referencing zarr_small's objects in place (Blob V2 external URIs)",
    "lance_ref_sharded": "LanceStore referencing zarr_sharded's objects in place (Blob V2 external URIs)",
}
# Reference layouts copy no chunk bytes: each row points at an object of this plain Zarr layout,
# which must be built first.
REFERENCED = {"lance_ref_small": "zarr_small", "lance_ref_sharded": "zarr_sharded"}


# The same Lance tables read through LanceDB Enterprise (scripts/enterprise.sh) instead of
# directly from object storage. Not built: they read what the lance_* layouts wrote.
ENTERPRISE_LAYOUTS = {
    "ent_big": "lance_big through the Enterprise REST API",
    "ent_small": "lance_small through the Enterprise REST API",
    "ent_tensor": "lance_tensor through the Enterprise REST API",
    "ent_sharded_blob": "lance_sharded_blob through the Enterprise REST API (blob byte ranges, bytes in Lance)",
    "ent_ref_small": "lance_ref_small through the Enterprise REST API (blob byte ranges)",
    "ent_ref_sharded": "lance_ref_sharded through the Enterprise REST API (blob byte ranges)",
}
ENTERPRISE_URL = os.environ.get("ENTERPRISE_URL", "http://127.0.0.1:47410")
ENTERPRISE_API_KEY = os.environ.get("ENTERPRISE_API_KEY", "localkey")


def build(layout: str, target: Target, cube: np.ndarray) -> dict:
    t0 = time.perf_counter()
    if layout.startswith("zarr_"):
        store = target.zarr_store(layout, read_only=False)
        write_zarr(store, cube, BIG if layout == "zarr_big" else SMALL, BIG if layout == "zarr_sharded" else None)
    elif layout.startswith("icechunk_"):
        repo = icechunk.Repository.create(target.icechunk_storage(layout))
        session = repo.writable_session("main")
        write_zarr(session.store, cube, SMALL, None)
        session.commit("initial write")
    elif layout == "lance_tensor":
        table = chip_rows(cube)
        ds = lance.write_dataset(table, target.lance_uri(layout), storage_options=target.lance_opts(),
                                 data_storage_version="2.2", max_rows_per_file=1024)
        ds.create_scalar_index("t", "BTREE")
    elif layout in REFERENCED:
        build_references(layout, target)
    else:
        enc = "blob" if layout.endswith("_blob") else "binary"
        store = LanceStore(target.lance_uri(layout), value_encoding=enc, storage_options=target.lance_opts())
        chunks = BIG if layout == "lance_big" else SMALL
        write_zarr(store, cube, chunks, BIG if layout == "lance_sharded_blob" else None)
        store.commit()
    return {"layout": layout, "write_s": round(time.perf_counter() - t0, 2)}


def s3_client(endpoint: str):
    import boto3

    return boto3.client("s3", endpoint_url=endpoint, aws_access_key_id=S3_KEY, aws_secret_access_key=S3_SECRET,
                        region_name=S3_REGION)


def s3_objects(endpoint: str, prefix: str) -> dict[str, int]:
    """Object key (relative to `prefix`) -> size, from a bucket listing."""
    sizes = {}
    for page in s3_client(endpoint).get_paginator("list_objects_v2").paginate(Bucket=S3_BUCKET, Prefix=prefix):
        sizes.update({o["Key"][len(prefix):]: o["Size"] for o in page.get("Contents", [])})
    return sizes


def s3_prefix(layout: str) -> str:
    if layout.startswith("zarr_"):
        return f"{layout}.zarr/"
    if layout.startswith("icechunk_"):
        return f"{layout}.icechunk/"
    return f"{LANCE_DB}/{layout}.lance/"


def build_references(layout: str, target: Target) -> None:
    """A `LanceStore` table whose values are external references to a plain Zarr layout's objects."""
    from lance.blob import Blob, blob_array

    zarr_layout = REFERENCED[layout]
    if target.name == "local":
        root = f"{ROOT}/{zarr_layout}.zarr"
        sizes = {os.path.relpath(os.path.join(dp, f), root): os.path.getsize(os.path.join(dp, f))
                 for dp, _, fns in os.walk(root) for f in fns}
        root = "file://" + root
    else:
        root = f"s3://{S3_BUCKET}/{zarr_layout}.zarr"
        sizes = s3_objects(target.endpoint, s3_prefix(zarr_layout))
        s3 = s3_client(target.endpoint)
        for key in s3_objects(target.endpoint, s3_prefix(layout)):  # a rebuild starts from nothing
            s3.delete_object(Bucket=S3_BUCKET, Key=s3_prefix(layout) + key)
    keys = sorted(sizes)
    rows = pa.table({"key": pa.array(keys, pa.string()),
                     "value": blob_array([Blob.from_uri(f"{root}/{k}", 0, sizes[k]) for k in keys])},
                    schema=pa.schema([pa.field("key", pa.string(), nullable=False), lance.blob_field("value")]))
    # The Zarr root is registered as a base so references may point into it; Lance's own files
    # still go to the dataset root.
    lance.write_dataset(rows, target.lance_uri(layout), storage_options=target.lance_opts(),
                        data_storage_version="2.2", mode="create",
                        initial_bases=[lance.DatasetBasePath(root, name="zarr")])


def open_array(layout: str, target: Target, counting: bool = False):
    from lance_zarr import CountingStore, LanceDBStore, LanceDBTensorStore, LanceTensorStore

    # Blob reads run in a thread pool; size it to Zarr's concurrency so it is not the limit.
    blob_threads = zarr.config.get("async.concurrency")
    if layout.startswith("ent_"):
        table, conn = "lance_" + layout[len("ent_"):], dict(database=LANCE_DB, host=ENTERPRISE_URL, api_key=ENTERPRISE_API_KEY)
        store = (LanceDBTensorStore(table, **conn) if layout == "ent_tensor"
                 else LanceDBStore(table, **conn, blob_read_threads=blob_threads))
    elif layout.startswith("zarr_"):
        store = target.zarr_store(layout, read_only=True)
    elif layout.startswith("icechunk_"):
        store = icechunk.Repository.open(target.icechunk_storage(layout)).readonly_session("main").store
    elif layout == "lance_tensor":
        store = LanceTensorStore(target.lance_uri(layout), storage_options=target.lance_opts())
    else:
        store = LanceStore(target.lance_uri(layout), read_only=True, storage_options=target.lance_opts(),
                           blob_read_threads=blob_threads)
    if counting:
        store = CountingStore(store)
    return zarr.open_array(store, path="reflectance", mode="r"), store


RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def save(name: str, obj) -> None:
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, f"{name}.json"), "w") as f:
        json.dump(obj, f, indent=2)


def reset_local(layout: str) -> None:
    for ext in (".zarr", ".icechunk", ".lance"):
        shutil.rmtree(f"{ROOT}/{layout}{ext}", ignore_errors=True)
