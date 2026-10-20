# Zarr store backed by Lance

This repo tests how far a Zarr store backed by Lance gets with features that exist today
(pylance 13.0.0, LanceDB 0.40.0, zarr-python 3.4.1, Icechunk 2.3.0). It started from a design
drafted in a conversation about a Zarr store to lancedb.

Everything below ran on one Mac (Apple silicon). The object-store numbers come from MinIO in
Docker with injected latency, not from S3. The caveats section says what that does and does
not show.

## Findings

- **A Zarr store over a Lance table works.** `LanceStore` keeps one Lance row per Zarr key. It
  passes zarr-python's own store test suite with a plain binary value column and with a Blob V2
  value column. xarray writes and reads datasets through it unchanged.
- **Lance versions and tags behave as Zarr snapshots.** Each `commit()` is one Lance version.
  A store opened at a version number or tag is a read-only view of that state.
- **Zarr sharding works inside Lance.** With a Blob V2 value column, the shard-index and chunk
  reads become blob byte-range reads. The Lance layout sends exactly the same requests and
  bytes to object storage as sharded Zarr on plain objects.
- **A Zarr view over decoded tensors works.** `LanceTensorStore` serves a Zarr array from a
  table with one decoded image chip per row, and the same rows answer SQL filters and vector
  search. It was the fastest layout for the AOI workload on local disk. On the object-store rig it sent far fewer
  requests than any other small-chunk layout but was slower than Icechunk, up to twice as slow
  on a full date, because Lance stores the chips uncompressed and it reads about 1.7 times the
  bytes.
- **On object storage, the chunk layout decides read cost, not the container.** On the rig,
  Lance with small chunks was level with Icechunk at Zarr's default concurrency and up to about
  a third slower at higher concurrency. On local disk it was faster. Lance with big chunks
  matched Zarr with big chunks. Lance does not make a poor chunk layout cheaper.
- **Lance sends fewer requests for the same chunks.** A Lance `take` merges adjacent rows into
  one request, so small-chunk Lance sent up to 68% fewer requests than Icechunk. On this rig
  that did not make it faster. On S3 it would lower request cost; whether it lowers latency
  needs a real S3 run.
- **Grid-cell filters plus vector search work in LanceDB.** H3 and S2 cell IDs stored as uint64
  with BTREE indexes turn "inside this parent cell" and "inside this S2 covering" into range
  filters. Both returned exactly the rows found by checking every cell in Python, including S2
  IDs above the signed 64-bit range, and the query plan shows the BTREE lookup feeding the vector
  search as its prefilter.
- **Through LanceDB Enterprise, a warm cache made small-chunk and tensor reads two to nine times
  faster than direct reads.** The small chunks sit in a `large_binary` column. The gain needs the
  reader to pin a table version; unpinned, each read pays an object-store round trip for a
  version check. Cold reads are slower than direct because the plan executor's disk cache
  fetches misses in 32 KiB requests; a larger `--disk-cache-page-size` fixes that today.
- **Writes through Enterprise work, including MemWAL.** With 4 concurrent writers, MemWAL was the
  fastest commit path and created no Lance version per commit. Its 16 MiB request cap surfaces as
  a misleading 503.
- **An existing plain Zarr store can be served through Lance without copying it.** A table of
  Zarr keys and Blob V2 external references to the store's objects is 0.07% of the Zarr bytes and
  reads correctly through `LanceStore` and through Enterprise. Read directly, it sends exactly the
  requests and bytes plain Zarr sends and runs close to it. With metadata stored inline, Lance
  versions and tags work as snapshots of the plain store when data is only appended; an in-place
  overwrite breaks old versions. Through Enterprise, referenced bytes bypass the cache: cold and
  warm reads take the same time, 1.3 to 2 times longer than direct.
- **Blob V2 reads through Enterprise are not cached yet**, whether the bytes are stored in Lance or
  referenced, because the query node serves blob byte ranges itself. ENT-3108 tracks serving them
  through the plan executor's cache and ENT-3100 batching them. Until then, chunks that need the
  cache belong in a `large_binary` column.
- **Two Lance gaps showed up, both with clear paths.** Format 2.2 compresses scalar, binary and
  string columns, but never fixed-size lists, the type a tensor or chip column uses;
  [lance-format/lance#5021](https://github.com/lance-format/lance/issues/5021) covers it.
  `Dataset.read_blob_ranges` re-reads blob descriptors on every call, which costs an extra round
  trip on short reads; no issue exists yet. Details are in the gaps section.

## What is in the repo

| Path | What it is |
|---|---|
| [lance_zarr/store.py](lance_zarr/store.py) | `LanceStore`: a full zarr-python v3 `Store` over a Lance dataset with `key` and `value` columns |
| [lance_zarr/tensor_store.py](lance_zarr/tensor_store.py) | `LanceTensorStore`: a read-only Zarr view over a table of decoded chips in a fixed-shape tensor column |
| [lance_zarr/remote.py](lance_zarr/remote.py) | `LanceDBStore` and `LanceDBTensorStore`: the same layouts read and written through LanceDB Enterprise's REST API |
| [lance_zarr/counting.py](lance_zarr/counting.py) | `CountingStore`: wraps any store and counts gets, ranged gets and bytes |
| [tests/](tests/) | zarr's store conformance suite against `LanceStore`, a tensor-view test, and a versioning smoke test |
| [demo/dual_table.py](demo/dual_table.py) | The Gemini dual-table design built for real: a per-pixel LanceDB table plus Zarr chunks in Lance |
| [demo/reference_zarr.py](demo/reference_zarr.py) | An existing plain Zarr store served through `LanceStore` by reference, plus re-indexing after an append |
| [bench/](bench/) | Layout builder, read workloads, report generator, and the diagnostics quoted below |
| [scripts/](scripts/) | MinIO and toxiproxy up and down, LanceDB Enterprise over the MinIO bucket, and a latency proxy that counts requests |
| [results/](results/) | Raw JSON and text output behind every number in this README |

## How `LanceStore` maps Zarr onto Lance

- **Rows.** One row per Zarr key, metadata documents and chunks alike. The key is the full
  Zarr key, such as `rgb_nir/c/3/0/1/2`, so time, band and chunk position never collide.
- **Reads.** zarr-python issues one `get` per chunk, with up to `async.concurrency` in flight.
  The store collects every `get` from the same event-loop tick and resolves them with one Lance
  `take`. A key-to-row map is built when the store opens, the same way Icechunk loads its
  manifest.
- **Byte ranges.** With a binary value column, the store reads the value and slices it. With a
  Blob V2 value column, every read is a blob byte range. When the store opens, it opens one
  `BlobFile` handle per row from the blob descriptors in a single read. Each handle knows its
  blob's location and size, so a read costs one request per range: `BlobFile.read_ranges` per
  blob, with blobs read in parallel.
- **Writes.** `set` and `delete` are buffered and written as one Lance version by `commit()`:
  a `merge_insert` on `key`, then one delete. `autocommit=True` commits on every call, which is
  what the conformance suite expects.
- **Snapshots.** `tag(name)` tags the current version. `LanceStore(uri, version="name")` opens it
  read-only.

Test status: `pytest tests` gives 129 passed and 26 skipped. The skipped tests cover optional
synchronous store methods that zarr-python does not require.

## `LanceTensorStore`: the Zarr API without Zarr chunks

The table has one row per (time, tile row, tile column), with the pixels in a fixed-shape tensor
column of shape (bands, 64, 64) and ordinary columns beside it, such as an embedding vector,
cloud fraction and acquisition time. The store synthesizes the Zarr metadata for an array of
shape (time, band, y, x) with one chip per chunk and the plain `bytes` codec. It serves each
chunk key from its row. Zarr and xarray assemble windows across chips. The same rows answer
`to_table(filter="cloud_frac < 0.1")` and LanceDB vector search without going through Zarr.

This is the closest thing here to "keep the Zarr API but not Zarr files". It is read-only.
Writing through the Zarr API into tensor rows is not implemented.

## Dual-table demo

[demo/dual_table.py](demo/dual_table.py) builds a 512 × 512 pixel region near Manaus, with 12
dates of 4-band imagery and a 128-d embedding per pixel. The region sits on S2 face 4, where
level-16 cell IDs exceed the signed 64-bit range.

- **Table 2** is an xarray Dataset (`rgb_nir` and an int8 `tessera` cube) written with
  `to_zarr` into a Blob V2 `LanceStore`, committed and tagged `demo-v1`.
- **Table 1** is a LanceDB table with one row per pixel: uint64 `h3` (resolution 13), `h3_r11`,
  `s2_l16`, year, global `y` and `x`, and a float32 embedding. It has an IVF_PQ index on the
  embedding and BTREE indexes on `h3`, `s2_l16` and `year`.

What it checks, from [results/dual_table.json](results/dual_table.json):

| Check | Result |
|---|---|
| Vector search, then xarray windows of 100 × 100 px × 12 dates around 5 hits, from the tagged snapshot | windows equal the source cube |
| H3 resolution-9 parent as one `BETWEEN` on the resolution-13 column | 1,094 rows; checking every cell in Python finds 1,094 |
| S2 covering of 4 cells as an OR of `BETWEEN` ranges, largest ID 10550813662019321855 | 50,234 rows; checking every cell in Python finds 50,234 |
| Both filters as vector-search prefilters | `ScalarIndexQuery` on the BTREE feeds the ANN search ([plans](results/dual_table_plans.txt)) |
| Pixels per H3 resolution-11 cell | mean 22, min 1, max 25 |
| Embeddings stored twice, as the design does | 134 MB raw float32 in Table 1, 34 MB raw int8 in Table 2 |

The H3 trick: an H3 index stores one 3-bit digit per resolution, so every descendant of a cell
at a finer resolution shares its prefix. Setting the resolution field to the child resolution
and filling the new digits with 0 gives the low end of one contiguous uint64 range. Filling
them with 6 gives the high end. S2 cells have the same property through `range_min` and
`range_max`. H3 containment is hierarchical: children cover their parent only approximately, so
"inside this parent" means H3's parent relation, not exact polygon containment. S2 coverings are
exact unions of cells.

Resolution 11 is about 21 pixels of 10 m per cell, so a row per resolution-11 cell is not a row
per pixel. Resolution 13 (about 44 m²) gives each pixel its own cell, which is what Table 1 uses.

## Benchmarks

### Data and layouts

A synthetic cube of 12 dates × 14 bands × 1024 × 1024 uint16, 352 MB raw: smooth fields plus
noise. Every layout that compresses uses Blosc with zstd level 5 and byte shuffle.

| Layout | What it stores | Size | Files |
|---|---|---|---|
| `zarr_big` | Zarr v3, 512 × 512 chunks with all 14 bands | 206 MB | 50 |
| `zarr_sharded` | Zarr v3, 64 × 64 chunks inside 512 × 512 shards | 210 MB | 50 |
| `icechunk_small` | Icechunk, 64 × 64 chunks | 210 MB | 3,079 |
| `lance_big` | `LanceStore`, binary column, 512 × 512 chunks | 206 MB | 4 |
| `lance_small` | `LanceStore`, binary column, 64 × 64 chunks | 210 MB | 4 |
| `lance_sharded_blob` | `LanceStore`, Blob V2 column, 64 × 64 chunks inside 512 × 512 shards | 210 MB | 52 |
| `lance_tensor` | `LanceTensorStore`, one 14 × 64 × 64 chip per row, uncompressed | 353 MB | 10 |

Each workload runs as a sequence of reads, the way an analysis loop issues them. The first,
counted pass checks every window against the source cube.

- **20 AOIs**: 20 windows of 100 × 100 px across all 12 dates and 14 bands.
- **128 patches**: 128 windows of 32 × 32 px, each on one date.
- **One full date**: all bands of one date.

In the tables, "gets" counts zarr-level store calls. The Lance stores batch those into fewer
Lance calls, which issue their own HTTP requests. Those are counted separately below.

### Local disk, warm page cache

| Layout | 20 AOIs | 128 patches | One full date |
|---|---|---|---|
| zarr_big | 0.31 s · 315 gets · 1340 MB | 0.72 s · 143 gets · 601 MB | 0.01 s · 7 gets · 17 MB |
| zarr_sharded | 0.32 s · 699 gets · 193 MB | 0.15 s · 290 gets · 36 MB | 0.03 s · 7 gets · 18 MB |
| icechunk_small | 0.44 s · 1359 gets · 93 MB | 0.18 s · 284 gets · 19 MB | 0.07 s · 259 gets · 18 MB |
| lance_big | 0.47 s · 315 gets · 1340 MB | 0.82 s · 143 gets · 601 MB | 0.01 s · 7 gets · 17 MB |
| lance_small | 0.26 s · 1359 gets · 93 MB | 0.12 s · 284 gets · 19 MB | 0.06 s · 259 gets · 18 MB |
| lance_sharded_blob | 0.28 s · 699 gets · 193 MB | 0.19 s · 290 gets · 36 MB | 0.03 s · 7 gets · 18 MB |
| lance_tensor | 0.13 s · 1359 gets · 156 MB | 0.12 s · 284 gets · 32 MB | 0.03 s · 259 gets · 29 MB |

Local timings moved by up to about a third between runs, so treat smaller differences as noise.
The tensor view is clearly fastest on the AOIs and level with small-chunk Lance on the other two
workloads. Its chunks are raw tensors, so there is no Blosc decoding step; how much of the gap
that explains was not measured separately. Small-chunk Lance is faster than Icechunk and close
to sharded Zarr. The big-chunk layouts read 1.3 GB to return 20 small windows.

### Object-store rig: MinIO with 20 ms added latency

MinIO runs in Docker. Reads go through toxiproxy, which adds 20 ± 5 ms to every response.
Writes go direct. The rig's own limits, from [results/diag_range_latency.txt](results/diag_range_latency.txt)
and [results/diag_rig_throughput.txt](results/diag_rig_throughput.txt):

| Rig measurement | Value |
|---|---|
| 64 KB range GET from a 210 MB object, direct | 3.7 ms |
| 64 KB range GET from a 0.1 MB object, direct | 1.8 ms |
| Total GET throughput, direct or through the proxy, 8 to 32 parallel | about 200 MB/s |

Docker Desktop's port forwarding caps total bandwidth at about 200 MB/s, far below what S3
gives an in-region client. Rows that read hundreds of MB are bandwidth-bound here and would be
faster on S3. Rows that read tens of MB are dominated by round trips, which this rig models
reasonably.

Zarr's default `async.concurrency` of 10:

| Layout | 20 AOIs | 128 patches | One full date |
|---|---|---|---|
| zarr_big | 7.09 s · 315 gets · 1340 MB | 6.82 s · 143 gets · 601 MB | 0.11 s · 7 gets · 17 MB |
| zarr_sharded | 2.71 s · 699 gets · 193 MB | 6.63 s · 290 gets · 36 MB | 0.16 s · 7 gets · 18 MB |
| icechunk_small | 3.70 s · 1359 gets · 93 MB | 3.42 s · 284 gets · 19 MB | 0.68 s · 259 gets · 18 MB |
| lance_big | 7.70 s · 315 gets · 1340 MB | 6.90 s · 143 gets · 601 MB | 0.12 s · 7 gets · 17 MB |
| lance_small | 3.94 s · 1359 gets · 93 MB | 3.59 s · 284 gets · 19 MB | 0.68 s · 259 gets · 18 MB |
| lance_sharded_blob | 3.31 s · 699 gets · 193 MB | 6.76 s · 290 gets · 36 MB | 0.15 s · 7 gets · 18 MB |
| lance_tensor | 4.59 s · 1359 gets · 156 MB | 3.75 s · 284 gets · 32 MB | 0.90 s · 259 gets · 29 MB |

`async.concurrency` raised to 64:

| Layout | 20 AOIs | 128 patches | One full date |
|---|---|---|---|
| zarr_big | 7.62 s · 315 gets · 1340 MB | 6.90 s · 143 gets · 601 MB | 0.11 s · 7 gets · 17 MB |
| zarr_sharded | 1.81 s · 699 gets · 193 MB | 6.58 s · 290 gets · 36 MB | 0.14 s · 7 gets · 18 MB |
| icechunk_small | 1.10 s · 1359 gets · 93 MB | 3.58 s · 284 gets · 19 MB | 0.16 s · 259 gets · 18 MB |
| lance_big | 7.51 s · 315 gets · 1340 MB | 6.95 s · 143 gets · 601 MB | 0.12 s · 7 gets · 17 MB |
| lance_small | 1.33 s · 1359 gets · 93 MB | 3.62 s · 284 gets · 19 MB | 0.22 s · 259 gets · 18 MB |
| lance_sharded_blob | 2.42 s · 699 gets · 193 MB | 6.69 s · 290 gets · 36 MB | 0.14 s · 7 gets · 18 MB |
| lance_tensor | 1.60 s · 1359 gets · 156 MB | 3.74 s · 284 gets · 32 MB | 0.31 s · 259 gets · 29 MB |

The `lance_sharded_blob` rows come from a rerun after the store started opening blob handles at
load ([results/run_s3_minio_c10_blob_rerun.json](results/run_s3_minio_c10_blob_rerun.json)). No
other layout uses that code.

### What the object-store numbers say

- **Big chunks are expensive in any container.** Both big-chunk layouts read 1.3 GB for 20
  small windows and took about 7 s, bound by this rig's bandwidth.
- **The patch workload counts round trips.** Icechunk, small-chunk Lance and the tensor view
  take one round trip per patch, about 3.5 s for 128 patches. Both sharded layouts take two,
  shard index then chunks, about 6.6 s.
- **Concurrency limits what a store can batch.** At Zarr's default of 10, a store never sees
  more than 10 chunk requests at once, so Lance has little to merge. At 64, small-chunk Lance
  went from 3.94 s to 1.33 s on the AOIs, and Icechunk went from 3.70 s to 1.10 s.
- **The tensor view trades bytes for requests.** It sent the fewest requests of any small-chunk
  layout but read about 1.7 times the bytes. On this bandwidth-limited rig the bytes won.
- **The blob-sharded layout matches sharded Zarr on I/O.** It sends the same requests and bytes,
  shown in the next table. Its remaining AOI gap of about 0.6 s is therefore on the client side,
  in the Python store or how its reads are scheduled.

### HTTP requests per workload

Counted by a proxy in front of MinIO, with opening costs excluded
([bench/diag_workload_requests.py](bench/diag_workload_requests.py)). Each cell shows the count
at concurrency 10, then at 64.

| Layout | 20 AOIs | 128 patches | One full date |
|---|---|---|---|
| zarr_sharded | 696 / 696 | 287 / 287 | 4 / 4 |
| lance_sharded_blob | 696 / 696 | 287 / 287 | 4 / 4 |
| icechunk_small | 1357 / 1357 | 282 / 282 | 257 / 257 |
| lance_small | 1144 / 819 | 212 / 212 | 204 / 83 |
| lance_tensor | 630 / 600 | 195 / 195 | 28 / 6 |

A Lance `take` of scattered rows sends one request per row, and those run in parallel. A take of
100 rows finished in 4 to 5 latency periods ([results/diag_requests.txt](results/diag_requests.txt)).
Rows that sit next to each other share a request, which is where the lower counts come from.

## Through LanceDB Enterprise

`LanceDBStore` and `LanceDBTensorStore` ([lance_zarr/remote.py](lance_zarr/remote.py)) serve the
same tables through a LanceDB Enterprise server instead of reading object storage directly.
[scripts/enterprise.sh](scripts/enterprise.sh) runs the `lancedb` CLI release
`2026-10-06.ffb7c5b7fb` as a query node plus one plan executor, with root `s3://bench`, reading
MinIO through the same 20 ms proxy. The plan executor has a 2 GiB memory page cache in front of a
disk page cache. No Agent runs, so nothing compacts or re-indexes the benchmark tables.

Reads are REST takes by row ID. The query node routes each take to the plan executor that owns
the data, which serves it from its cache or fetches it.

### Reads: cold, disk-warm and memory-warm cache

Zarr concurrency 64, from [results/run_enterprise_c64.json](results/run_enterprise_c64.json).
"Cold" restarts the plan executor with both cache tiers empty. "Disk" restarts it with only the
memory tier empty. "Memory" repeats the reads with both warm. "Direct" is the same table read
from MinIO by the client with no Enterprise in the path.

| Table | Workload | Cold | Disk | Memory | Direct |
|---|---|---|---|---|---|
| small chunks | 20 AOIs | 3.35 s | 0.62 s | 0.51 s | 1.28 s |
| small chunks | 128 patches | 4.02 s | 0.55 s | 0.38 s | 3.44 s |
| small chunks | one full date | 0.55 s | 0.11 s | 0.09 s | 0.21 s |
| tensor | 20 AOIs | 2.70 s | 0.73 s | 0.65 s | 1.56 s |
| tensor | 128 patches | 4.20 s | 0.61 s | 0.46 s | 3.65 s |
| tensor | one full date | 1.30 s | 0.23 s | 0.13 s | 0.30 s |
| big chunks | 20 AOIs | 8.52 s | 5.65 s | 5.30 s | 7.17 s |
| big chunks | 128 patches | 14.04 s | 3.00 s | 2.82 s | 6.77 s |
| big chunks | one full date | 0.59 s | 0.10 s | 0.10 s | 0.11 s |

- **A warm cache is the gain.** The 128 sequential patch reads drop from about 3.5 s to about
  0.4 s, because each read no longer pays an object-store round trip. The AOIs and full dates
  drop by two to three times against direct reads.
- **Big chunks stay expensive.** Every byte crosses the REST API to the client. The big-chunk
  table reads 1.3 GB for 20 small windows, which takes about 5 s even from memory here.
- **Cold reads are slower than direct reads**, for the reason in the next section.

### Pin a version, or every read pays a storage round trip

Under the server's default strong consistency, a request against a table's latest version first
asks object storage whether a newer version exists. With a warm cache that check is most of a
read's latency ([bench/diag_enterprise_take.py](bench/diag_enterprise_take.py)):

| Warm take | Unpinned, strong consistency | `--weak-read-consistency-interval-seconds 600` | Pinned version, strong consistency |
|---|---|---|---|
| 1 row | 25.7 ms | 2.5 ms | 2.1 ms |
| 64 rows | 41.9 ms | 15.8 ms | 17.4 ms |

The stores pin a version when they open, as a Zarr snapshot reader would anyway, so the table
above was measured that way.

### The disk cache splits cold reads into 32 KiB requests

On a miss, the memory tier merges a run of missing pages into one wide read from the disk tier.
The disk tier then fetches each missing page with its own object-store request
(`load_missing_pages` in `sophon-caching/src/object_store/paging.rs`, unchanged on Sophon main as
of 2026-10-19). Its default page is 32 KiB. Counted with a proxy in front of MinIO
([bench/diag_enterprise_cold.py](bench/diag_enterprise_cold.py)):

| Cold pass | Disk cache, 32 KiB pages | Disk cache, 1 MiB pages | Memory cache only | Direct Lance |
|---|---|---|---|---|
| Small chunks, 20 AOIs | 3,128 requests, 2.11 s | 171 requests, 1.50 s | 621 requests, 1.70 s | 752 requests, 1.40 s |
| Tensor, one full date | 902 requests, 1.21 s | 32 requests, 0.38 s | 8 requests, 0.40 s | 6 requests, 0.22 s |

`--disk-cache-page-size` fixes the request count today, at the cost of reading more bytes: 1 MiB
pages read 179 MB for the AOIs against 104 MB with 32 KiB pages. Merging contiguous misses in the
disk tier, as the memory tier already does, would fix it without the extra bytes.

### Writes, and MemWAL

[bench/run_enterprise_writes.py](bench/run_enterprise_writes.py) writes the whole cube with
64 × 64 chunks, committing after each of the 12 dates. With 4 writers, writer i takes dates i,
i + 4 and i + 8 in its own process. Every path writes through the latency proxy, and every run is
read back and compared with the cube.

| Path | 1 writer | 4 writers | Median commit, 1 writer | Lance versions after |
|---|---|---|---|---|
| Sharded Zarr on MinIO, no commit step | 4.04 s | 2.13 s | – | – |
| `LanceStore` straight to MinIO | 7.10 s | 4.60 s | 0.44 s | 13 |
| `LanceDBStore`, REST merge_insert | 9.79 s | 4.52 s | 0.63 s | 14 |
| `LanceDBStore`, REST merge_insert through MemWAL | 8.44 s | 3.84 s | 0.41 s | 2 |

All four paths verified with 4 concurrent writers, and no writer hit a commit conflict. MemWAL
writes add no Lance versions; the data sits in the WAL shard until a flush. Three details:

- **The WAL needs an unenforced primary key.** The SDK refuses `set_unenforced_primary_key` on a
  remote table, and the store declares it in the `key` field's metadata when it creates the
  table. Then `set_lsm_write_spec` installs the WAL. `ALTER TABLE <t> SET WAL` does the same from
  SQL. The CLI's server always runs a local WAL shard.
- **The WAL shard caps a request body at 16 MiB.** One date's commit is about 17.5 MB. The
  oversized request comes back as a 503, "wal shard ... unreachable", which the client retries
  three times. The store now splits MemWAL merges into requests under 8 MiB.
- **A take by row ID cannot read unflushed WAL data.** On a WAL table the store reads with
  `key IN (...)` filter queries instead, which the LSM read path answers from base plus
  memtables.

### Blob byte ranges through Enterprise are uncached and one per request

A REST take on a Blob V2 column returns the blob descriptors, not the bytes. Byte ranges do exist:
`GET /v1/table/{id}/blob/{column}/{row_id}/bytes` with an HTTP Range header, which the SDK uses for
remote `BlobFile.read_range`. Each request serves one range of one row on the query node: a
descriptor take, then a read from object storage, with no plan executor and so no page cache.
Repeated 2 KB and 64 KB range reads took 46 ms each, the same as the first, against 2.1 ms for a
warm pinned take of a binary row. The SDK's remote `fetch_blob_ranges`, which would batch ranges,
raises "not supported". So sharded Zarr in a blob column reads through Enterprise one range at a
time at storage latency; the binary and tensor tables use the cached take path. Batching is filed
as ENT-3100 and caching as ENT-3108.

## Plain Zarr referenced in place

A Blob V2 value can be an external reference: a URI plus an offset and size, stored without
copying the bytes. A `LanceStore` table whose rows hold each Zarr key and a reference to the
object a plain Zarr writer already produced serves that store through the Zarr API, with the
chunk bytes left where they are. The references must point under a base path registered on the
table (`initial_bases=[lance.DatasetBasePath(zarr_root, name="zarr")]`), or the write is refused.

Chunk files are written once, so they are referenced. The `zarr.json` metadata documents are
rewritten in place on every append, so [demo/reference_zarr.py](demo/reference_zarr.py) stores
their bytes inline in the same blob column. On local disk
([results/reference_zarr.json](results/reference_zarr.json)), for a chunked and a sharded Zarr v3
store:

| Check | Chunked | Sharded |
|---|---|---|
| Reads through Lance equal the source | yes | yes |
| Lance table for 13.5 MB of Zarr | 25 KB | 15 KB |
| Rows re-indexed after appending one date (listing by size and mtime; on S3 the ETag) | 20 | 5 |
| Latest version after the append | all 7 dates, exact | all 7 dates, exact |
| Version from before the append, by number and by tag | original 6 dates, exact | original 6 dates, exact |
| That old version after overwriting one date in place | read fails: short read | read fails: checksum mismatch |

So Lance versions and tags give snapshots over plain Zarr, without Icechunk and without copying
chunks, for data that is only appended. Lance versions the references, not the referenced files:
an in-place overwrite breaks the versions that pointed at the old bytes. Data that is reprocessed
in place needs new file names per write, a copy into Lance, or Icechunk.

On the object-store rig, `build_references` in [bench/common.py](bench/common.py) lists a plain
Zarr layout's objects and writes the reference table
([results/build_s3_minio_ref.json](results/build_s3_minio_ref.json)):

| Layout | References | Lance table | Files |
|---|---|---|---|
| `lance_ref_small` | `zarr_small`: 3,074 objects, 210 MB, one per 64 × 64 chunk | 151 KB | 3 |
| `lance_ref_sharded` | `zarr_sharded`: 50 objects, 210 MB | 6 KB | 3 |

### Reads, same session as the baselines

Median seconds at Zarr concurrency 10, then 64, every window verified against the source
([results/run_s3_c10_minio_ref.json](results/run_s3_c10_minio_ref.json),
[results/run_s3_c64_minio_ref.json](results/run_s3_c64_minio_ref.json)):

| Layout | Bytes live in | 20 AOIs | 128 patches | One full date |
|---|---|---|---|---|
| `zarr_small` | plain objects | 3.59 / 1.00 | 3.38 / 3.40 | 0.64 / 0.14 |
| `icechunk_small` | Icechunk | 3.58 / 1.05 | 3.40 / 3.47 | 0.66 / 0.17 |
| `lance_small` | Lance | 3.82 / 1.30 | 3.50 / 3.57 | 0.65 / 0.20 |
| `lance_ref_small` | `zarr_small`'s objects | 3.83 / 1.28 | 3.47 / 3.42 | 0.69 / 0.22 |
| `zarr_sharded` | plain objects | 2.66 / 1.77 | 6.54 / 6.58 | 0.13 / 0.12 |
| `lance_sharded_blob` | Lance | 3.25 / 2.12 | 6.59 / 6.54 | 0.15 / 0.14 |
| `lance_ref_sharded` | `zarr_sharded`'s objects | 3.34 / 2.06 | 6.56 / 6.49 | 0.16 / 0.15 |

HTTP requests at concurrency 64, counted by the proxy
([results/diag_workload_requests_ref.txt](results/diag_workload_requests_ref.txt)):

| Layout | 20 AOIs | 128 patches | One full date |
|---|---|---|---|
| `zarr_small` | 1,356 | 281 | 256 |
| `lance_ref_small` | 1,356 | 281 | 256 |
| `lance_small` | 750 | 203 | 79 |
| `zarr_sharded` | 696 | 287 | 4 |
| `lance_ref_sharded` | 696 | 287 | 4 |

- **A referenced layout sends exactly the requests and bytes of the Zarr it references.** Each
  chunk is its own object, so Lance's merging of adjacent rows, which cuts `lance_small` to 750
  requests, does not apply. The remaining time gap at concurrency 64, up to about 0.3 s, is the
  same as for `lance_small` and sits on the client side, in the Python store.
- **Blob reads now use as many threads as Zarr's concurrency.** The store used a fixed 16-thread
  pool before. At concurrency 64 that took `lance_sharded_blob` from 2.42 s to 2.12 s on the AOIs,
  against 1.77 s for sharded Zarr.

### Through LanceDB Enterprise

`LanceDBStore` reads a blob table through remote `BlobFile` handles, opened for every row when the
store opens: one REST request per byte range, each served by the query node. Concurrency 64, with
the in-Lance `lance_small` table run in the same session for contrast
([results/run_enterprise_c64_ref_small.json](results/run_enterprise_c64_ref_small.json),
[results/run_enterprise_c64_ref_sharded.json](results/run_enterprise_c64_ref_sharded.json),
[results/run_enterprise_c64_ref_baseline.json](results/run_enterprise_c64_ref_baseline.json)):

| Table | Workload | Cold | Disk | Memory | Direct |
|---|---|---|---|---|---|
| `lance_ref_small` | 20 AOIs | 2.07 s | 2.01 s | 1.97 s | 1.27 s |
| `lance_ref_small` | 128 patches | 6.41 s | 6.41 s | 6.44 s | 3.45 s |
| `lance_ref_small` | one full date | 0.32 s | 0.35 s | 0.33 s | 0.22 s |
| `lance_ref_sharded` | 20 AOIs | 3.95 s | 2.95 s | 2.95 s | 2.03 s |
| `lance_ref_sharded` | 128 patches | 12.90 s | 12.74 s | 12.84 s | 6.46 s |
| `lance_ref_sharded` | one full date | 0.18 s | 0.16 s | 0.16 s | 0.13 s |
| `lance_small`, bytes in Lance | 20 AOIs | 2.17 s | 0.60 s | 0.53 s | 1.31 s |
| `lance_small`, bytes in Lance | 128 patches | 3.95 s | 0.42 s | 0.35 s | 3.52 s |
| `lance_small`, bytes in Lance | one full date | 0.55 s | 0.11 s | 0.10 s | 0.20 s |

- **Referenced bytes get no cache.** Cold, disk-warm and memory-warm reads take the same time,
  because blob byte ranges go through the query node's blob route, not the plan executor.
- **Each range costs more than a direct read.** A request does a descriptor take and then the
  object read, so sequential reads such as the patches, two per window for shards, take about twice
  as long as direct. Cached blob ranges (ENT-3108) and batched ones (ENT-3100) would cover
  referenced data too.
- **Opening is slow for many rows.** Opening a remote handle for each of `lance_ref_small`'s 3,074
  rows took about 18 s. Opening is outside the timings above.

### Is Blob V2 cached through Enterprise, when the bytes are stored in Lance?

No. [bench/diag_enterprise_blob_cache.py](bench/diag_enterprise_blob_cache.py) runs Enterprise
with a counting latency proxy between both processes and MinIO, restarts the plan executor with
empty caches, and reads each workload twice
([results/diag_enterprise_blob_cache.txt](results/diag_enterprise_blob_cache.txt)):

| Table | Workload | Cold pass | Warm pass |
|---|---|---|---|
| `lance_small`, binary column | 128 patches | 920 requests, 4.49 s | 0 requests, 0.37 s |
| `lance_small`, binary column | one full date | 538 requests, 0.68 s | 0 requests, 0.10 s |
| `lance_sharded_blob`, Blob V2, bytes in Lance | 128 patches | 574 requests, 12.81 s | 574 requests, 12.62 s |
| `lance_sharded_blob`, Blob V2, bytes in Lance | one full date | 8 requests, 0.13 s | 8 requests, 0.13 s |
| `lance_ref_sharded`, Blob V2, external references | 128 patches | 574 requests, 12.97 s | 574 requests, 12.66 s |
| `lance_ref_sharded`, Blob V2, external references | one full date | 8 requests, 0.13 s | 8 requests, 0.13 s |

The query node serves blob byte ranges itself, so the plan executor's cache never sees them,
whether the blob bytes live in Lance or elsewhere. Each range costs two object-store requests,
287 ranges sending 574, which is why these reads take about twice as long as direct. Only
binary (`large_binary`) and tensor columns get the cache today. ENT-3108 tracks routing both blob
routes through the plan executor so Blob V2 reads are cached too; until it lands, chunks that need
the cache belong in a `large_binary` column.

## Lance gaps and development paths

### Fixed-size lists are not compressed in format 2.2

All Lance data here is file format 2.2, the pylance 13 default and the format Blob V2 requires.
Format 2.2 does compress: scalar columns get bitpacking or zstd, and binary or string values get
per-value compression. Fixed-size lists, the type a tensor or chip column uses, are the
exception. In the 2.2 strategy (`rust/lance-file/src/versions/v2_2/compression.rs`, identical at
the v13.0.0 tag):

- Full-zip, used for wide values such as a whole chip, stores fixed-width and fixed-size-list
  values raw (`try_raw_per_value`). Compression applies only to variable-width values.
- Mini-block, used for narrow values, wraps general compression only around plain fixed-width
  columns (`finalize_miniblock_compressor`). A fixed-size list goes to a raw encoder.
- Mini-block also rejects values wider than about 8 KiB, so a chip cannot use it at all.
- The 2.3 strategy handles fixed-size lists the same way.

Measured on 200 smooth 14 × 64 × 64 uint16 chips, 114,688 bytes each, always the same pixels
([bench/diag_wide_fsl_compression.py](bench/diag_wide_fsl_compression.py)):

| How the pixels are stored | Size relative to raw |
|---|---|
| Blosc zstd with byte shuffle, per chip, as Zarr writes them | 0.59 |
| One chip per row as a fixed-size list, default or zstd requested | 1.00 |
| One chip per row as a fixed-size list, mini-block with zstd | write fails: value too wide for mini-block |
| One chip per row as binary, default or zstd requested | 0.78 |
| 8-value fixed-size list per row, default or zstd requested | 1.00 |
| One uint16 per row, default or zstd requested | 0.76 |

Storing chips as binary is a workaround available today, at 0.78 of raw. Blosc does better
because it shuffles bytes before zstd, which suits uint16 rasters, and Lance's per-value path
has no shuffle step. The development path is per-value compression for fixed-width and
fixed-size-list data, tracked in
[lance-format/lance#5021](https://github.com/lance-format/lance/issues/5021), ideally with a
byte-shuffle option. Until then the tensor view stores about 1.7 times the bytes of
Blosc-compressed Zarr.

### `read_blob_ranges` re-reads blob descriptors on every call

Each call first fetches the descriptor page for the requested rows, then the payload ranges.
That is two sequential requests for a read that needs one
([bench/diag_blob_paths.py](bench/diag_blob_paths.py)):

| Call | Requests per call | Median time |
|---|---|---|
| `Dataset.read_blob_ranges`, one suffix range | 2 | 46.5 ms |
| `Dataset.read_blob_ranges`, 4 ranges in one blob | 5 | 50.0 ms |
| `BlobFile.read_range` on a handle opened earlier, one range | 1 | 23.7 ms |
| `BlobFile.read_ranges` on a handle opened earlier, 4 ranges | 4 | 25.9 ms |

Zarr sharding makes two dependent reads per window, shard index then chunks, so this turned two
round trips into four. With `read_blob_ranges`, the patch workload took 12.8 s against 6.6 s for
sharded Zarr. With handles opened at load it takes 6.8 s. The batch API still wins for bulk reads
that merge many ranges across blobs, as measured in
[lance-format/lance#7864](https://github.com/lance-format/lance/pull/7864). Development paths:
cache descriptors in the dataset's metadata cache, or let callers pass descriptors they already
hold. No issue exists for this yet.

### Fixed in this repo, no Lance change needed

- **Blob sizes.** The first version looked up sizes with `take_blobs` before every ranged batch,
  which added another sequential request: the patch workload took 19.0 s. The handles opened at
  load carry the sizes. Scanning a Blob V2 column also returns its descriptors (kind, position,
  size) without payloads.
- **Thread splitting.** An earlier run on moto, a Python S3 mock, suggested Lance sent per-row
  requests almost one at a time, so the store split batches across 16 threads. moto serves a
  64 KB range from a 210 MB object in about 22 ms against 2 to 3 ms for small objects, which is
  what slowed those reads. On MinIO, Lance's requests run in parallel, and the thread split
  changed the small-chunk and tensor workloads by under 5%, so it was removed. The moto results
  are kept as `results/run_s3_moto_*_INVALID.json`.

## Caveats

- The data is synthetic. Real reflectance and real embeddings compress differently.
- One machine, MinIO in Docker, injected latency, and a 200 MB/s bandwidth ceiling. This is not
  an S3 measurement.
- Reads only, one client. No concurrent writers, no commit conflicts, no compaction.
- `LanceTensorStore` is read-only and stores chips uncompressed.
- The Enterprise runs put the client, query node and plan executor on one machine, so no network
  sits between the client and the server, and one plan executor holds every page.
- The Blob V2 store opens a handle for every row at load. That suits shard-sized rows, a few
  thousand per array. A store with millions of rows would open handles lazily or bound them.
- The dual-table demo covers one 512 × 512 region. It shows the mechanisms work. It does not
  show how they scale to the billions of rows a per-pixel table would hold.

## Reproduce

```bash
uv sync
uv run pytest -q tests                        # conformance and tensor view
uv run python tests/smoke.py                  # versions and tags
uv run python demo/dual_table.py              # writes results/dual_table.json

cd bench
uv run python build.py --target local
uv run python run.py --target local --repeats 3 --concurrency 10

bash ../scripts/s3_up.sh                      # MinIO on :19000, toxiproxy on :19001
uv run python build.py --target s3
uv run python run.py --target s3 --repeats 2 --concurrency 10
uv run python run.py --target s3 --repeats 2 --concurrency 64
uv run python report.py build_local.json run_local_c10.json run_s3_c10.json run_s3_c64.json

# request counts: a counting proxy on :19002 in front of MinIO
PROXY_COUNT_FILE=../results/proxy_counts_minio.txt \
  uv run python ../scripts/latency_proxy.py --listen 127.0.0.1:19002 --upstream 127.0.0.1:19000 --jitter 0 &
PROXY_COUNT_FILE=../results/proxy_counts_minio.txt uv run python diag_workload_requests.py \
  http://localhost:19002 zarr_sharded lance_sharded_blob icechunk_small lance_small lance_tensor

# through LanceDB Enterprise (needs ~/dev/lancedb-cli-helpers for the lancedb binary)
bash ../scripts/enterprise.sh start           # REST on :47410, database zarr
uv run python run_enterprise.py --concurrency 64
uv run python run_enterprise_writes.py --writers 1 4

# plain Zarr referenced in place
uv run python ../demo/reference_zarr.py      # local: correctness, size, re-index after an append
uv run python build.py --target s3 --layouts zarr_small lance_ref_small lance_ref_sharded --tag _minio_ref
uv run python run.py --target s3 --repeats 2 --concurrency 64 --tag _minio_ref --layouts \
  icechunk_small zarr_small lance_small lance_ref_small zarr_sharded lance_sharded_blob lance_ref_sharded
uv run python run_enterprise.py --layouts ent_ref_small lance_ref_small --tag _ref_small
bash ../scripts/enterprise.sh stop
bash ../scripts/s3_down.sh
```

## Next steps

- Run the object-store workloads on real S3 from an in-region instance, where bandwidth stops
  hiding the request-count differences.
- Compress the tensor view and re-measure: store chips in a binary column (0.78 of raw today),
  store Blosc-compressed chips in a binary column, or use a Lance per-value codec for
  fixed-size lists once #5021 lands.
- File the `read_blob_ranges` descriptor issue on lance-format/lance.
- Repeat with real Sentinel-2 tiles and real TESSERA embeddings.
- Add a write path to the tensor view, so Zarr writes land as tensor rows.
- Run the referenced layouts against real S3, and re-measure them through Enterprise once blob
  byte ranges are cached and batched (ENT-3108, ENT-3100).
