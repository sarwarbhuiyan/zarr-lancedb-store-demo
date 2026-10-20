"""Write the cube through each write path with one writer and with several, then verify.

Paths, every one writing through the latency proxy:
  zarr_sharded   Zarr v3 on MinIO through obstore: PUTs only, no commit step
  lance_direct   LanceStore straight to MinIO: one Lance commit (merge_insert) per date
  ent_merge      LanceDBStore through Enterprise: one REST merge_insert per date
  ent_wal        LanceDBStore through Enterprise's MemWAL: one REST merge_insert per date, into the WAL

Each run creates a fresh array (64 x 64 chunks, Blosc), then --writers processes write it: writer i
takes dates i, i + W, ... and commits after each date. Afterwards the whole array is read back
through a fresh read-only store and compared with the cube. Needs scripts/s3_up.sh and
scripts/enterprise.sh running.
"""

import argparse
import multiprocessing as mp
import statistics
import time
import uuid

import numpy as np
import zarr

from common import (B, COMPRESSOR, ENTERPRISE_API_KEY, ENTERPRISE_URL, H, S3_PROXY, SMALL, T, W, Target,
                    save)
from lance_zarr import LanceDBStore, LanceStore
from workloads import cube

WRITES_DB = "zarr_writes"
PATHS = ["zarr_sharded", "lance_direct", "ent_merge", "ent_wal"]
SHARDS = (1, B, 512, 512)
target = Target("s3", S3_PROXY)


def open_store(path: str, name: str, read_only: bool):
    if path == "zarr_sharded":
        return target.zarr_store(f"{WRITES_DB}/{name}", read_only=read_only)
    if path == "lance_direct":
        uri = f"s3://bench/{WRITES_DB}/{name}.lance"
        return LanceStore(uri, read_only=read_only, storage_options=target.lance_opts())
    return LanceDBStore(name, database=WRITES_DB, host=ENTERPRISE_URL, api_key=ENTERPRISE_API_KEY,
                        read_only=read_only, use_lsm=(path == "ent_wal"))


def commit(store) -> None:
    if hasattr(store, "commit"):
        store.commit()


def create_array(path: str, name: str) -> None:
    store = open_store(path, name, read_only=False)
    zarr.create_array(store, name="reflectance", shape=(T, B, H, W), chunks=SMALL, dtype="uint16",
                      fill_value=0, compressors=COMPRESSOR, shards=SHARDS if path == "zarr_sharded" else None)
    commit(store)


def write_dates(path: str, name: str, dates: list[int]) -> dict:
    """Runs in a worker process: write each date, commit after each one."""
    zarr.config.set({"async.concurrency": 64})
    store = open_store(path, name, read_only=False)
    arr = zarr.open_array(store, path="reflectance", mode="r+")
    commit_s, error = [], None
    try:
        for t in dates:
            arr[t] = cube[t]
            t0 = time.perf_counter()
            commit(store)
            commit_s.append(time.perf_counter() - t0)
    except Exception as e:  # report and keep the other writers' results
        error = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
    return {"commit_s": commit_s, "error": error}


def verify(path: str, name: str) -> bool:
    zarr.config.set({"async.concurrency": 64})
    arr = zarr.open_array(open_store(path, name, read_only=True), path="reflectance", mode="r")
    return bool(np.array_equal(arr[:], cube))


def run(path: str, writers: int) -> dict:
    name = f"{path}_{writers}w_{uuid.uuid4().hex[:6]}"
    create_array(path, name)
    slices = [list(range(i, T, writers)) for i in range(writers)]
    t0 = time.perf_counter()
    with mp.get_context("spawn").Pool(writers) as pool:
        results = pool.starmap(write_dates, [(path, name, s) for s in slices])
    wall = time.perf_counter() - t0
    commits = [c for r in results for c in r["commit_s"]]
    errors = [r["error"] for r in results if r["error"]]
    row = {
        "path": path, "writers": writers, "wall_s": round(wall, 2),
        "dates_written": len(commits), "commit_median_s": round(statistics.median(commits), 3) if commits else None,
        "errors": errors, "table": name,
    }
    t0 = time.perf_counter()
    row["verified"] = verify(path, name) if not errors else False
    row["verify_read_s"] = round(time.perf_counter() - t0, 2)
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", nargs="*", default=PATHS)
    ap.add_argument("--writers", nargs="*", type=int, default=[1, 4])
    args = ap.parse_args()
    rows = []
    for writers in args.writers:
        for path in args.paths:
            row = run(path, writers)
            print(row, flush=True)
            rows.append(row)
    save("run_enterprise_writes", rows)
