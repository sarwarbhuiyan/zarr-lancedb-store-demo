"""Round trips per ranged blob read: read_blob_ranges vs reusable BlobFile handles.

Run with scripts/latency_proxy.py in front of the S3 server; pass its URL.
"""

import os
import statistics
import sys
import time

import lance

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import S3_PROXY, Target  # noqa: E402

COUNTS = os.environ.get("PROXY_COUNT_FILE", "results/proxy_counts_minio.txt")
target = Target("s3", sys.argv[1] if len(sys.argv) > 1 else S3_PROXY)


def counts():
    try:
        return int(open(COUNTS).read().split()[0])
    except (FileNotFoundError, ValueError, IndexError):
        return 0


ds = lance.dataset(target.lance_uri("lance_sharded_blob"), storage_options=target.lance_opts())
desc = ds.to_table(columns=["value"]).column("value").combine_chunks()
sizes = (desc.storage if hasattr(desc, "storage") else desc).field("size").to_pylist()
rows = [i for i, s in enumerate(sizes) if s > 1 << 20][:8]


def measure(name, fn, n=8):
    fn(0)
    reqs, times = [], []
    for j in range(1, n + 1):
        r0, t0 = counts(), time.perf_counter()
        fn(j)
        times.append((time.perf_counter() - t0) * 1000)
        reqs.append(counts() - r0)
    print(f"{name:44} median {statistics.median(times):6.1f} ms, requests per call {statistics.median(reqs):.0f}")


def suffix(j):
    row = rows[j % len(rows)]
    return row, sizes[row] - 2048, 2048


measure("read_blob_ranges, 1 suffix range", lambda j: ds.read_blob_ranges("value", [suffix(j)], selector="indices"))
measure("read_blob_ranges, 4 ranges in 1 blob",
        lambda j: ds.read_blob_ranges("value", [(rows[j % len(rows)], k * 300_000, 65536) for k in range(4)], selector="indices"))
handles = dict(zip(rows, ds.take_blobs("value", indices=rows)))
measure("cached BlobFile.read_range, 1 suffix range", lambda j: handles[suffix(j)[0]].read_range(suffix(j)[1], 2048))
measure("cached BlobFile.read_ranges, 4 ranges",
        lambda j: handles[rows[j % len(rows)]].read_ranges([(k * 300_000, 65536) for k in range(4)]))
