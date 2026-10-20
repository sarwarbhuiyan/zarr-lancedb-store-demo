"""Gemini's dual-table design, built with real Lance and LanceDB features and checked end to end.

Table 1 (LanceDB `earth_pixels`): one row per 10 m pixel per year, with H3 and S2 cell IDs
as uint64 columns, the pixel's global (y, x) position, and a 128-d embedding.
Table 2 (LanceStore): the Zarr chunks of an xarray Dataset (`rgb_nir` imagery and the
per-pixel `tessera` embedding cube), one Lance row per Zarr key, read back through xarray.

The demo then runs the hand-off the design describes: vector search in Table 1, then an
xarray window read from Table 2 around each hit; plus grid-cell range filters (H3 containment,
S2 covering with IDs above the signed 64-bit range) combined with vector search.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import time

import h3
import lancedb
from lancedb.index import BTree, IvfPq
import numpy as np
import pyarrow as pa
import s2sphere
import xarray as xr
from zarr.codecs import BloscCodec

from lance_zarr import LanceStore

ROOT = os.path.expanduser("~/dev/zarr-lancedb-store-demo/data")
OUT = f"{ROOT}/dual"
N = 512                      # region: 512 x 512 pixels of 10 m = 5.12 km square
PIX_M = 10.0
LAT0, LON0 = -3.1, -60.0     # near Manaus: S2 face 4, so S2 cell IDs exceed int64 max
RGBN = [1, 2, 3, 7]          # bands standing in for B02, B03, B04, B08 in the synthetic stack
EMB = 128
H3_RES, S2_LEVEL = 13, 16
timings: dict[str, float] = {}


def tick(name: str, t0: float) -> None:
    timings[name] = round(time.perf_counter() - t0, 3)
    print(f"  {name}: {timings[name]} s", flush=True)


def pixel_latlng(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lat = LAT0 - (y + 0.5) * PIX_M / 111_320.0
    lng = LON0 + (x + 0.5) * PIX_M / (111_320.0 * math.cos(math.radians(LAT0)))
    return lat, lng


def uses_scalar_index(plan: str) -> bool:
    return "ScalarIndexQuery" in plan or "MaterializeIndex" in plan


def h3_children_range(parent: int, child_res: int) -> tuple[int, int]:
    """[lo, hi] of all descendants of `parent` at `child_res`, by padding the H3 digits."""
    parent_res = (parent >> 52) & 0xF
    base = (parent & ~(0xF << 52)) | (child_res << 52)
    lo = hi = base
    for r in range(parent_res + 1, child_res + 1):
        shift = 3 * (15 - r)
        lo = (lo & ~(0x7 << shift)) | (0 << shift)
        hi = (hi & ~(0x7 << shift)) | (6 << shift)
    return lo, hi


def dir_mb(path: str) -> float:
    return round(sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(path) for f in fs) / 1e6, 1)


def pixel_embeddings(cube: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A unit-norm EMB-d float32 embedding per pixel, (N, N, EMB): a random projection of each
    pixel's per-band mean and standard deviation over time, plus noise. A stand-in for TESSERA."""
    stack = np.asarray(cube, dtype=np.float32) / 10000.0          # (T, B, N, N)
    feats = np.concatenate([stack.mean(0), stack.std(0)], axis=0)  # (2B, N, N): level + seasonality
    proj = rng.normal(size=(feats.shape[0], EMB)).astype(np.float32)
    n = cube.shape[-1]
    emb = np.einsum("fyx,fe->yxe", feats, proj) + rng.normal(0, 0.02, (n, n, EMB)).astype(np.float32)
    return emb / np.linalg.norm(emb, axis=-1, keepdims=True)


def main() -> None:
    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(OUT)
    cube = np.load(f"{ROOT}/cube.npy", mmap_mode="r")[:, :, :N, :N]
    T = cube.shape[0]
    rng = np.random.default_rng(7)

    print("embeddings")
    t0 = time.perf_counter()
    emb = pixel_embeddings(cube, rng)
    tick("embeddings", t0)

    # ---- Table 2: an xarray Dataset written through the Zarr API into a Lance dataset ----
    print("table 2: xarray -> Zarr -> LanceStore")
    t0 = time.perf_counter()
    ys, xs = np.arange(N), np.arange(N)
    ds = xr.Dataset(
        {
            "rgb_nir": (("time", "band", "y", "x"), np.ascontiguousarray(cube[:, RGBN])),
            "tessera": (("y", "x", "emb"), np.round(emb * 127).astype(np.int8)),
        },
        coords={
            "time": np.array([f"2025-{m:02d}-15" for m in range(1, T + 1)], dtype="datetime64[ns]"),
            "band": ["B02", "B03", "B04", "B08"],
            "y": 9_657_000.0 - (ys + 0.5) * PIX_M,     # UTM-like northing, metres
            "x": 600_000.0 + (xs + 0.5) * PIX_M,       # UTM-like easting, metres
        },
    )
    blosc = BloscCodec(cname="zstd", clevel=5, shuffle="shuffle")
    store = LanceStore(f"{OUT}/earth_chunks.lance", value_encoding="blob")
    ds.to_zarr(
        store, zarr_format=3, consolidated=False, mode="w",
        encoding={
            "rgb_nir": {"chunks": (1, 4, 256, 256), "compressors": blosc},
            "tessera": {"chunks": (256, 256, EMB), "compressors": blosc},
        },
    )
    v_chunks = store.commit()
    store.tag("demo-v1")
    tick("table2_write", t0)

    # ---- Table 1: per-pixel rows with grid cell IDs and the embedding (LanceDB) ----
    print("table 1: per-pixel LanceDB table")
    t0 = time.perf_counter()
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    lat, lng = pixel_latlng(yy.ravel(), xx.ravel())
    h3_13 = np.fromiter((h3.str_to_int(h3.latlng_to_cell(a, b, H3_RES)) for a, b in zip(lat, lng)), np.uint64, lat.size)
    h3_11 = np.fromiter((h3.str_to_int(h3.cell_to_parent(h3.int_to_str(int(c)), 11)) for c in h3_13), np.uint64, lat.size)
    s2_16 = np.fromiter(
        (s2sphere.CellId.from_lat_lng(s2sphere.LatLng.from_degrees(a, b)).parent(S2_LEVEL).id() for a, b in zip(lat, lng)),
        np.uint64, lat.size,
    )
    tick("cell_ids", t0)
    t0 = time.perf_counter()
    pixels = pa.table({
        "h3": pa.array(h3_13, pa.uint64()),
        "h3_r11": pa.array(h3_11, pa.uint64()),
        "s2_l16": pa.array(s2_16, pa.uint64()),
        "year": pa.array(np.full(lat.size, 2025, np.int16)),
        "y": pa.array(yy.ravel().astype(np.int32)),
        "x": pa.array(xx.ravel().astype(np.int32)),
        "tessera_embedding": pa.FixedSizeListArray.from_arrays(pa.array(emb.reshape(-1)), EMB),
    })
    db = lancedb.connect(f"{OUT}/lancedb")
    tbl = db.create_table("earth_pixels", data=pixels, mode="overwrite")
    tbl.create_index("tessera_embedding", config=IvfPq(distance_type="cosine", num_partitions=64, num_sub_vectors=16))
    for col in ("h3", "s2_l16", "year"):
        tbl.create_index(col, config=BTree())
    tick("table1_write_and_index", t0)

    # ---- the hand-off: vector search in Table 1, xarray windows from Table 2 ----
    print("query: vector search -> xarray windows")
    zds = xr.open_zarr(LanceStore(f"{OUT}/earth_chunks.lance", version="demo-v1"), consolidated=False, zarr_format=3)
    qy, qx = 300, 200
    qvec = emb[qy, qx]
    t0 = time.perf_counter()
    hits = (tbl.search(qvec, vector_column_name="tessera_embedding").where("year = 2025", prefilter=True)
            .limit(5).select(["y", "x", "h3", "_distance"]).to_arrow())
    tick("vector_search_top5", t0)
    t0 = time.perf_counter()
    for y, x in zip(hits["y"].to_pylist(), hits["x"].to_pylist()):
        y0, x0 = max(y - 50, 0), max(x - 50, 0)
        window = zds["rgb_nir"].isel(y=slice(y0, y + 50), x=slice(x0, x + 50)).values
        assert np.array_equal(window, cube[:, RGBN, y0:y + 50, x0:x + 50]), "window mismatch"
    tick("xarray_windows_5x100px_12dates", t0)
    nearest = list(zip(hits["y"].to_pylist(), hits["x"].to_pylist()))
    print(f"  query pixel ({qy},{qx}); nearest hits {nearest}")

    # ---- H3 containment as a uint64 BTREE range, combined with vector search ----
    print("query: H3 parent containment as one range + vector search")
    q_cell = int(h3_13[qy * N + qx])
    parent = h3.str_to_int(h3.cell_to_parent(h3.int_to_str(q_cell), 9))
    lo, hi = h3_children_range(parent, H3_RES)
    expect = int(sum(h3.cell_to_parent(h3.int_to_str(int(c)), 9) == h3.int_to_str(parent) for c in h3_13))
    got = tbl.count_rows(f"h3 BETWEEN {lo} AND {hi}")
    t0 = time.perf_counter()
    in_parent = (tbl.search(qvec, vector_column_name="tessera_embedding")
                 .where(f"h3 BETWEEN {lo} AND {hi}", prefilter=True).limit(10).to_arrow())
    tick("vector_search_in_h3_parent", t0)
    h3_plan = (tbl.search(qvec, vector_column_name="tessera_embedding")
               .where(f"h3 BETWEEN {lo} AND {hi}", prefilter=True).limit(10).explain_plan(verbose=True))
    print(f"  res-9 parent {h3.int_to_str(parent)}: range rows {got}, brute-force rows {expect}, "
          f"top hits all inside: {set(in_parent['h3'].to_pylist()) <= set(range(lo, hi + 1))}")

    # ---- S2 covering on face 4 (IDs above int64 max) as a union of ranges ----
    print("query: S2 covering with IDs > int64 max + vector search")
    lat_lo, lat_hi = float(lat.min() + 0.01), float(lat.min() + 0.025)
    lng_lo, lng_hi = float(lng.min() + 0.01), float(lng.min() + 0.03)
    rect = s2sphere.LatLngRect.from_point_pair(s2sphere.LatLng.from_degrees(lat_lo, lng_lo), s2sphere.LatLng.from_degrees(lat_hi, lng_hi))
    coverer = s2sphere.RegionCoverer()
    coverer.min_level, coverer.max_level, coverer.max_cells = 10, S2_LEVEL, 8
    cells = coverer.get_covering(rect)
    ranges = [(c.range_min().id(), c.range_max().id()) for c in cells]
    pred = " OR ".join(f"(s2_l16 BETWEEN {a} AND {b})" for a, b in ranges)
    expect_s2 = int(sum(any(a <= int(v) <= b for a, b in ranges) for v in s2_16))
    got_s2 = tbl.count_rows(pred)
    t0 = time.perf_counter()
    s2_hits = (tbl.search(qvec, vector_column_name="tessera_embedding").where(pred, prefilter=True).limit(10).to_arrow())
    tick("vector_search_in_s2_covering", t0)
    s2_plan = (tbl.search(qvec, vector_column_name="tessera_embedding").where(pred, prefilter=True)
               .limit(10).explain_plan(verbose=True))
    print(f"  covering: {len(cells)} cells, max id {max(b for _, b in ranges)} (> int64 max: "
          f"{max(b for _, b in ranges) > 2**63 - 1}); range rows {got_s2}, brute-force rows {expect_s2}, hits {s2_hits.num_rows}")

    # ---- what the design's numbers look like ----
    per_r11 = np.unique(h3_11, return_counts=True)[1]
    sizes = {
        "table1_pixels_lancedb_mb": dir_mb(f"{OUT}/lancedb"),
        "table2_zarr_chunks_lance_mb": dir_mb(f"{OUT}/earth_chunks.lance"),
        "tessera_in_table2_raw_mb": round(N * N * EMB / 1e6, 1),
        "tessera_in_table1_raw_mb": round(N * N * EMB * 4 / 1e6, 1),
    }
    report = {
        "region_pixels": N * N,
        "table2_lance_version": v_chunks,
        "h3_res11_pixels_per_cell": {"mean": round(float(per_r11.mean()), 1), "min": int(per_r11.min()), "max": int(per_r11.max())},
        "h3_res13_distinct_cells": int(np.unique(h3_13).size),
        "h3_range_rows": got, "h3_bruteforce_rows": expect,
        "s2_cover_cells": len(cells), "s2_range_rows": got_s2, "s2_bruteforce_rows": expect_s2,
        "s2_ids_exceed_int64": bool(max(b for _, b in ranges) > 2**63 - 1),
        "h3_plan_uses_scalar_index": uses_scalar_index(h3_plan),
        "s2_plan_uses_scalar_index": uses_scalar_index(s2_plan),
        "sizes": sizes, "timings_s": timings,
    }
    os.makedirs("results", exist_ok=True)
    with open("results/dual_table.json", "w") as f:
        json.dump(report, f, indent=2)
    with open("results/dual_table_plans.txt", "w") as f:
        f.write(f"# H3 parent range\n{h3_plan}\n\n# S2 covering ranges\n{s2_plan}\n")
    print(json.dumps(report, indent=2))
    assert got == expect and got_s2 == expect_s2, "range filters disagree with brute force"


if __name__ == "__main__":
    main()
