"""What one chunk costs to read as an array grows to millions of chunks: Icechunk and Lance.

Icechunk's documentation warns that past about a million chunks "requesting even a single
chunk will require downloading the entire manifest", unless the manifest is split. This
builds one uint8 array per size, 100 dates of (1, 32, 32) chunks with no compression, and
writes the same chunk bytes to:

- an Icechunk repository, committed unsplit on `main` and re-split with `rewrite_manifests`
  on three branches: `split_time` (every 10 dates), `split_space` (10 x 10 chunk tiles, all
  dates) and `split_all` (every 10 chunks on every axis);
- a Lance table with one row per Zarr key and a BTREE on `key`, read through `LanceStore`
  with `key_lookup="map"` (every key read at open) and `key_lookup="index"` (BTREE lookups).

Each read runs in a fresh process, so no manifest, key map or index page is cached, through
scripts/latency_proxy.py, which adds a latency to every response and counts requests and
bytes. Opening and reading are counted separately.

`run-enterprise` reads the same Lance tables through LanceDB Enterprise, started with
S3_ENDPOINT at the counting proxy so the counts are what the query node and plan executor
send to object storage. Each workload runs once with the plan executor's caches emptied and
again from a new client process with them warm, the case of a new worker or agent process.

Usage (from bench/):
  python manifest_scale.py build 10000 100000 1000000
  python manifest_scale.py run --proxy http://127.0.0.1:19002 10000 100000 1000000
  S3_ENDPOINT=http://127.0.0.1:19002 ../scripts/enterprise.sh start
  python manifest_scale.py run-enterprise 10000 100000 1000000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
import subprocess
import sys
import time

import icechunk
import lance
import numpy as np
import pyarrow as pa
import zarr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    ENTERPRISE_API_KEY, ENTERPRISE_URL, LANCE_DB, S3_BUCKET, S3_DIRECT, Target, s3_client, s3_objects, save,
)

from lance_zarr import LanceDBStore, LanceStore  # noqa: E402
from lance_zarr.store import _schema  # noqa: E402

DATES, CHUNK = 100, 32
CHUNK_BYTES = CHUNK * CHUNK
RANDOM_CHUNKS = 100
REPEATS = 3
COUNTS = os.environ.get("PROXY_COUNT_FILE", "../results/proxy_counts_manifest_scale.txt")

Axis, AnyAxis = icechunk.ManifestSplitDimCondition.Axis, icechunk.ManifestSplitDimCondition.Any
SPLITS = {
    "split_time": {Axis(0): 10},
    "split_space": {Axis(1): 10, Axis(2): 10},
    "split_all": {AnyAxis(): 10},
}
LAYOUTS = ["icechunk", *(f"icechunk_{b}" for b in SPLITS), "lance_map", "lance_index"]
ENTERPRISE_LAYOUTS = ["enterprise_map", "enterprise_index"]
ENTERPRISE_SH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "enterprise.sh")
WORKLOADS = ["first chunk", f"{RANDOM_CHUNKS} random chunks", "one pixel, all dates"]


def grid(n_chunks: int) -> int:
    """Chunks per spatial side, so DATES * side**2 is close to n_chunks."""
    return round(math.sqrt(n_chunks / DATES))


def layout_name(n_chunks: int) -> str:
    return f"mscale_{n_chunks}"


def chunk_offsets(t: int, side: int) -> np.ndarray:
    y, x = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
    return (t * 7919 + y * 104729 + x * 1299709) % 256


def date_blocks(t: int, side: int) -> np.ndarray:
    """(side, side, CHUNK, CHUNK) chunk contents for date t; every chunk differs."""
    base = np.arange(CHUNK_BYTES).reshape(CHUNK, CHUNK)
    return ((base[None, None] + chunk_offsets(t, side)[:, :, None, None]) % 256).astype(np.uint8)


def expected_chunk(t: int, y: int, x: int) -> np.ndarray:
    off = (t * 7919 + y * 104729 + x * 1299709) % 256
    return ((np.arange(CHUNK_BYTES).reshape(CHUNK, CHUNK) + off) % 256).astype(np.uint8)


def create_array(store, side: int):
    root = zarr.create_group(store=store, zarr_format=3, overwrite=True)
    return root.create_array(
        "arr", shape=(DATES, side * CHUNK, side * CHUNK), chunks=(1, CHUNK, CHUNK), dtype="uint8",
        compressors=None, filters=None, fill_value=0,
    )


def delete_prefix(prefix: str) -> None:
    s3 = s3_client(S3_DIRECT)
    keys = list(s3_objects(S3_DIRECT, prefix))
    for i in range(0, len(keys), 1000):
        s3.delete_objects(Bucket=S3_BUCKET, Delete={"Objects": [{"Key": prefix + k} for k in keys[i:i + 1000]]})


# ---- build ------------------------------------------------------------------------


def build_icechunk(n_chunks: int, side: int) -> dict:
    target, layout = Target("s3", S3_DIRECT), layout_name(n_chunks)
    delete_prefix(f"{layout}.icechunk/")
    storage = target.icechunk_storage(layout)
    t0 = time.perf_counter()
    repo = icechunk.Repository.create(storage, config=icechunk.RepositoryConfig(inline_chunk_threshold_bytes=0))
    session = repo.writable_session("main")
    arr = create_array(session.store, side)
    for t in range(DATES):
        arr[t] = date_blocks(t, side).transpose(0, 2, 1, 3).reshape(side * CHUNK, side * CHUNK)
    main = session.commit("write")
    write_s = time.perf_counter() - t0
    for branch, sizes in SPLITS.items():
        repo.create_branch(branch, snapshot_id=main)
        config = icechunk.RepositoryConfig(
            manifest=icechunk.ManifestConfig(
                splitting=icechunk.ManifestSplittingConfig.from_dict({icechunk.ManifestSplitCondition.AnyArray(): sizes})
            )
        )
        icechunk.Repository.open(storage, config=config).rewrite_manifests(f"split: {branch}", branch=branch)
    objects = s3_objects(S3_DIRECT, f"{layout}.icechunk/")
    manifests = {k: v for k, v in objects.items() if k.startswith("manifests/")}
    return {
        "write_s": round(write_s, 1), "objects": len(objects),
        "manifest_files": len(manifests), "manifest_bytes": sum(manifests.values()),
    }


def build_lance(n_chunks: int, side: int) -> dict:
    target, layout = Target("s3", S3_DIRECT), layout_name(n_chunks)
    delete_prefix(f"zarr/{layout}.lance/")
    schema = _schema("binary")
    meta = zarr.storage.MemoryStore()
    create_array(meta, side)
    meta_rows = {k: v.to_bytes() for k, v in meta._store_dict.items()}

    def batches():
        yield pa.record_batch([pa.array(list(meta_rows), pa.string()), pa.array(list(meta_rows.values()), pa.large_binary())],
                              schema=schema)
        keys_yx = [f"{y}/{x}" for y in range(side) for x in range(side)]
        offsets = pa.py_buffer(np.arange(0, side * side * CHUNK_BYTES + 1, CHUNK_BYTES, dtype=np.int64).tobytes())
        for t in range(DATES):
            values = pa.Array.from_buffers(
                pa.large_binary(), side * side, [None, offsets, pa.py_buffer(date_blocks(t, side).tobytes())]
            )
            yield pa.record_batch([pa.array([f"arr/c/{t}/{k}" for k in keys_yx], pa.string()), values], schema=schema)

    t0 = time.perf_counter()
    uri = target.lance_uri(layout)
    lance.write_dataset(pa.RecordBatchReader.from_batches(schema, batches()), uri, schema=schema,
                        storage_options=target.lance_opts(), data_storage_version="2.2")
    write_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    ds = lance.dataset(uri, storage_options=target.lance_opts())
    ds.create_scalar_index("key", "BTREE")
    index_s = time.perf_counter() - t0
    ds = lance.dataset(uri, storage_options=target.lance_opts())
    plan = ds.scanner(columns=["key", "value"], filter="key IN ('arr/c/0/0/0', 'arr/c/1/0/0')").explain_plan()
    objects = s3_objects(S3_DIRECT, f"zarr/{layout}.lance/")
    index = {k: v for k, v in objects.items() if k.startswith("_indices/")}
    return {
        "write_s": round(write_s, 1), "index_s": round(index_s, 1), "rows": ds.count_rows(),
        "fragments": len(ds.get_fragments()), "index_files": len(index), "index_bytes": sum(index.values()),
        "uses_btree": "ScalarIndexQuery" in plan,
    }


def build(sizes: list[int], lance_only: bool) -> None:
    zarr.config.set({"async.concurrency": 256})
    name = "build_manifest_scale_lance_only" if lance_only else "build_manifest_scale"
    out = {}
    for n in sizes:
        side = grid(n)
        print(f"building {DATES * side * side:,} chunks ({DATES} x {side} x {side})", flush=True)
        out[str(n)] = {"chunks": DATES * side * side, "lance": build_lance(n, side)}
        if not lance_only:
            out[str(n)]["icechunk"] = build_icechunk(n, side)
        print(json.dumps(out[str(n)]), flush=True)
    save(name, out)


# ---- one cold read (runs in its own process) --------------------------------------


def workload_selections(n_chunks: int, side: int, workload: str) -> list[tuple]:
    rng = np.random.default_rng(n_chunks)
    cells = [(int(rng.integers(DATES)), int(rng.integers(side)), int(rng.integers(side))) for _ in range(RANDOM_CHUNKS)]
    if workload == "first chunk":
        cells = cells[:1]
    if workload == "one pixel, all dates":
        _, y, x = cells[0]
        return [("series", y, x)]
    return [("chunk", t, y, x) for t, y, x in cells]


def counts() -> tuple[int, int]:
    time.sleep(0.2)  # the proxy writes its count file as bytes pass; let the last write land
    try:
        r, b = open(COUNTS).read().split()
        return int(r), int(b)
    except (FileNotFoundError, ValueError):
        return 0, 0


async def open_array(layout: str, n_chunks: int, endpoint: str):
    target = Target("s3", endpoint)
    if layout.startswith("icechunk"):
        branch = layout.removeprefix("icechunk").lstrip("_") or "main"
        repo = icechunk.Repository.open(target.icechunk_storage(layout_name(n_chunks)))
        store = repo.readonly_session(branch=branch).store
    elif layout.startswith("enterprise"):
        store = LanceDBStore(layout_name(n_chunks), database=LANCE_DB, host=ENTERPRISE_URL, api_key=ENTERPRISE_API_KEY,
                             key_lookup=layout.removeprefix("enterprise_"))
    else:
        store = LanceStore(target.lance_uri(layout_name(n_chunks)), read_only=True,
                           storage_options=target.lance_opts(), key_lookup=layout.removeprefix("lance_"))
    return await zarr.api.asynchronous.open_array(store=store, path="arr", zarr_format=3)


async def read(arr, selections: list[tuple]) -> bool:
    if selections[0][0] == "series":
        _, y, x = selections[0]
        got = await arr.getitem((slice(None), y * CHUNK, x * CHUNK))
        return all(int(got[t]) == int(expected_chunk(t, y, x)[0, 0]) for t in range(DATES))
    sels = [(t, slice(y * CHUNK, (y + 1) * CHUNK), slice(x * CHUNK, (x + 1) * CHUNK)) for _, t, y, x in selections]
    got = await asyncio.gather(*(arr.getitem(s) for s in sels))
    return all(np.array_equal(g, expected_chunk(t, y, x)) for g, (_, t, y, x) in zip(got, selections))


async def one(layout: str, n_chunks: int, workload: str, endpoint: str) -> dict:
    zarr.config.set({"async.concurrency": 64})
    selections = workload_selections(n_chunks, grid(n_chunks), workload)
    r0, b0 = counts()
    t0 = time.perf_counter()
    arr = await open_array(layout, n_chunks, endpoint)
    open_s = time.perf_counter() - t0
    r1, b1 = counts()
    t0 = time.perf_counter()
    ok = await read(arr, selections)
    read_s = time.perf_counter() - t0
    r2, b2 = counts()
    return {
        "layout": layout, "chunks": DATES * grid(n_chunks) ** 2, "workload": workload, "ok": ok,
        "open_s": round(open_s, 3), "read_s": round(read_s, 3),
        "open_requests": r1 - r0, "open_bytes": b1 - b0, "read_requests": r2 - r1, "read_bytes": b2 - b1,
    }


# ---- run all, each read in a fresh process ----------------------------------------


def run(sizes: list[int], proxy: str, layouts: list[str], name: str) -> None:
    rows = []
    print(f"{'chunks':>10} {'layout':22} {'workload':22} {'open s':>7} {'read s':>7} {'total s':>7} "
          f"{'requests':>9} {'MB':>7}", flush=True)
    for n in sizes:
        for layout in layouts:
            for workload in WORKLOADS:
                reps = []
                for _ in range(REPEATS):
                    reps.append(one_process(layout, n, workload, proxy))
                med = sorted(reps, key=lambda r: r["open_s"] + r["read_s"])[len(reps) // 2]
                med["total_s_runs"] = [round(r["open_s"] + r["read_s"], 3) for r in reps]
                med["ok"] = all(r["ok"] for r in reps)
                rows.append(med)
                req = med["open_requests"] + med["read_requests"]
                mb = (med["open_bytes"] + med["read_bytes"]) / 1e6
                print(f"{med['chunks']:>10,} {layout:22} {workload:22} {med['open_s']:7.2f} {med['read_s']:7.2f} "
                      f"{statistics.median(med['total_s_runs']):7.2f} {req:9} {mb:7.2f}"
                      f"{'' if med['ok'] else '  WRONG DATA'}", flush=True)
    save(name, rows)


def one_process(layout: str, n: int, workload: str, endpoint: str) -> dict:
    p = subprocess.run([sys.executable, __file__, "one", layout, str(n), workload, "--endpoint", endpoint],
                       capture_output=True, text=True, check=True)
    return json.loads(p.stdout.strip().splitlines()[-1])


def run_enterprise(sizes: list[int], layouts: list[str], name: str) -> None:
    rows = []
    print(f"{'chunks':>10} {'layout':18} {'workload':22} {'cache':5} {'open s':>7} {'read s':>7} "
          f"{'requests':>9} {'MB':>7}", flush=True)
    for n in sizes:
        for layout in layouts:
            for workload in WORKLOADS:
                subprocess.run(["bash", ENTERPRISE_SH, "restart-pe", "--cold"], check=True, stdout=subprocess.DEVNULL)
                for cache in ("cold", "warm"):
                    row = {**one_process(layout, n, workload, "unused"), "cache": cache}
                    rows.append(row)
                    req = row["open_requests"] + row["read_requests"]
                    mb = (row["open_bytes"] + row["read_bytes"]) / 1e6
                    print(f"{row['chunks']:>10,} {layout:18} {workload:22} {cache:5} {row['open_s']:7.2f} "
                          f"{row['read_s']:7.2f} {req:9} {mb:7.2f}{'' if row['ok'] else '  WRONG DATA'}", flush=True)
    save(name, rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--lance-only", action="store_true", help="skip Icechunk (its writes are one object per chunk)")
    b.add_argument("sizes", nargs="+", type=int)
    r = sub.add_parser("run")
    r.add_argument("--proxy", required=True)
    r.add_argument("--layouts", default=",".join(LAYOUTS))
    r.add_argument("--name", default="manifest_scale", help="results file name")
    r.add_argument("sizes", nargs="+", type=int)
    e = sub.add_parser("run-enterprise")
    e.add_argument("--layouts", default=",".join(ENTERPRISE_LAYOUTS))
    e.add_argument("--name", default="manifest_scale_enterprise", help="results file name")
    e.add_argument("sizes", nargs="+", type=int)
    o = sub.add_parser("one")
    o.add_argument("layout")
    o.add_argument("n_chunks", type=int)
    o.add_argument("workload")
    o.add_argument("--endpoint", required=True)
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.sizes, a.lance_only)
    elif a.cmd == "run":
        run(a.sizes, a.proxy, a.layouts.split(","), a.name)
    elif a.cmd == "run-enterprise":
        run_enterprise(a.sizes, a.layouts.split(","), a.name)
    else:
        print(json.dumps(asyncio.run(one(a.layout, a.n_chunks, a.workload, a.endpoint))))
