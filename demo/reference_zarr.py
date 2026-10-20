"""Serve an existing plain Zarr store through Lance without copying its chunks, with snapshots.

Each Lance row holds one Zarr key. Chunk files become external Blob V2 references (URI, offset,
size) to the objects the Zarr writer already produced; metadata documents (`zarr.json`) are small
and rewritten in place on every append, so their bytes are stored inline in Lance. `LanceStore`
reads both kinds of row the same way, so xarray opens the store unchanged.

Checks, for a chunked and a sharded Zarr v3 store:
1. every array read through Lance equals the source, and the Lance table is a small fraction of
   the Zarr bytes,
2. after appending a date to the plain Zarr store, a re-index rewrites only new or changed keys
   (found from object size and mtime; on S3, the ETag), with no Icechunk involved,
3. the version from before the append, opened by number or by tag, still reads the original
   arrays, while the latest version reads the appended ones,
4. after overwriting a date in place in the plain Zarr store, the old version no longer reads the
   original bytes: Lance versions the references, not the referenced files.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import xarray as xr
import zarr
from lance.blob import Blob, blob_array

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lance_zarr import LanceStore  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "runs" / "reference_zarr"
RESULTS = Path(__file__).resolve().parent.parent / "results" / "reference_zarr.json"
DATES, BANDS, SIDE = 6, 4, 512
BEFORE_APPEND = "before-append"
SCHEMA = pa.schema([
    pa.field("key", pa.string(), nullable=False),
    lance.blob_field("value"),
    pa.field("size", pa.int64()),
    pa.field("mtime_ns", pa.int64()),
])


def cube(dates: int, seed: int = 0) -> xr.Dataset:
    rng = np.random.default_rng(seed)
    data = rng.integers(0, 10_000, size=(dates, BANDS, SIDE, SIDE), dtype=np.uint16)
    return xr.Dataset({"rgb_nir": (("time", "band", "y", "x"), data)},
                      coords={"time": np.arange(dates)})


def listing(zarr_dir: Path) -> dict[str, tuple[int, int]]:
    """Zarr key -> (size, mtime_ns), the way an object-store listing returns size and ETag."""
    return {p.relative_to(zarr_dir).as_posix(): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in sorted(zarr_dir.rglob("*")) if p.is_file()}


def is_metadata(key: str) -> bool:
    return key.rsplit("/", 1)[-1] == "zarr.json"


def reference_rows(zarr_dir: Path, keys: list[str], objects: dict[str, tuple[int, int]]) -> pa.Table:
    # Metadata is rewritten in place on append, so a reference would let old versions see the new
    # shape; its bytes go inline. Chunk files are written once and are referenced where they are.
    blobs = [Blob.from_bytes((zarr_dir / k).read_bytes()) if is_metadata(k)
             else Blob.from_uri((zarr_dir / k).resolve().as_uri(), 0, objects[k][0]) for k in keys]
    return pa.table({
        "key": pa.array(keys, pa.string()),
        "value": blob_array(blobs),
        "size": pa.array([objects[k][0] for k in keys], pa.int64()),
        "mtime_ns": pa.array([objects[k][1] for k in keys], pa.int64()),
    }, schema=SCHEMA)


def reindex(zarr_dir: Path, lance_uri: str) -> dict:
    """Add or update the rows whose object is new or changed; one Lance version per call."""
    objects = listing(zarr_dir)
    try:
        ds = lance.dataset(lance_uri)
        known = ds.to_table(columns=["key", "size", "mtime_ns"]).to_pylist()
        seen = {r["key"]: (r["size"], r["mtime_ns"]) for r in known}
    except (FileNotFoundError, ValueError):
        ds, seen = None, {}
    changed = [k for k, v in objects.items() if seen.get(k) != v]
    if changed:
        rows = reference_rows(zarr_dir, changed, objects)
        if ds is None:
            # The Zarr root becomes a registered base: blob references may point into it, while
            # Lance's own data files still go to the dataset root.
            zarr_base = lance.DatasetBasePath(zarr_dir.resolve().as_uri(), name="zarr")
            lance.write_dataset(rows, lance_uri, schema=SCHEMA, data_storage_version="2.2",
                                initial_bases=[zarr_base])
        else:
            ds.merge_insert("key").when_matched_update_all().when_not_matched_insert_all().execute(rows)
    return {"objects": len(objects), "rows_written": len(changed),
            "metadata_rows_written": sum(is_metadata(k) for k in changed),
            "version": lance.dataset(lance_uri).version}


def du(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def read(lance_uri: str, version: int | str | None = None) -> xr.Dataset:
    store = LanceStore(lance_uri, read_only=True, version=version)
    return xr.open_zarr(store, zarr_format=3, consolidated=False)


def same(got: xr.Dataset, expected: xr.Dataset) -> bool:
    return got["rgb_nir"].shape == expected["rgb_nir"].shape and bool(
        np.array_equal(got["rgb_nir"].values, expected["rgb_nir"].values))


def old_version_outcome(lance_uri: str, version: int, expected: xr.Dataset) -> str:
    """What reading an old version returns once its referenced files have been overwritten.

    The reference keeps the old size, so a shorter new file fails the read and a longer one
    returns a prefix of the new bytes.
    """
    try:
        return "equals original" if same(read(lance_uri, version), expected) else "different bytes"
    except (OSError, ValueError, RuntimeError) as e:
        return f"read error: {str(e).splitlines()[0][:160]}"


def run(layout: str, encoding: dict) -> dict:
    zarr_dir, lance_uri = OUT / f"{layout}.zarr", str(OUT / f"{layout}.lance")
    first = cube(DATES)
    first.to_zarr(zarr_dir, zarr_format=3, consolidated=False, mode="w", encoding={"rgb_nir": encoding})

    t0 = time.perf_counter()
    initial = reindex(zarr_dir, lance_uri)
    initial["seconds"] = round(time.perf_counter() - t0, 3)
    lance.dataset(lance_uri).tags.create(BEFORE_APPEND, initial["version"])
    reads_equal = same(read(lance_uri), first)

    extra = cube(1, seed=1).assign_coords(time=[DATES])
    extra.to_zarr(zarr_dir, zarr_format=3, consolidated=False, append_dim="time")
    appended = xr.concat([first, extra], dim="time")
    t0 = time.perf_counter()
    after_append = reindex(zarr_dir, lance_uri)
    after_append["seconds"] = round(time.perf_counter() - t0, 3)
    old_by_version = read(lance_uri, initial["version"])
    snapshot = {
        "latest_equals_appended": same(read(lance_uri), appended),
        "old_version_dates": int(old_by_version["rgb_nir"].shape[0]),
        "old_version_equals_original": same(old_by_version, first),
        "old_tag_equals_original": same(read(lance_uri, BEFORE_APPEND), first),
    }

    # Overwrite date 0 in place, the way a reprocessing job would, then re-index.
    zarr.open_array(zarr_dir / "rgb_nir", mode="r+")[0] = cube(1, seed=2)["rgb_nir"].values[0]
    after_overwrite = reindex(zarr_dir, lance_uri)
    in_place = {"reindex": after_overwrite,
                "old_version_read": old_version_outcome(lance_uri, initial["version"], first)}

    return {
        "layout": layout,
        "zarr_bytes": du(zarr_dir),
        "lance_bytes": du(Path(lance_uri)),
        "initial_index": initial,
        "reads_equal_source": reads_equal,
        "after_appending_one_date": after_append,
        "snapshot_after_append": snapshot,
        "after_overwriting_one_date_in_place": in_place,
    }


def main() -> None:
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True)
    results = [
        run("chunked", {"chunks": (1, BANDS, 128, 128)}),
        run("sharded", {"chunks": (1, BANDS, 64, 64), "shards": (1, BANDS, SIDE, SIDE)}),
    ]
    RESULTS.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
