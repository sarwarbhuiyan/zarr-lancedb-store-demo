"""Count S3 requests and wall time for Lance takes of N scattered rows.

Run scripts/latency_proxy.py in front of the S3 server and pass its URL; the proxy writes the
response and byte counts this script reads.
"""

import os
import sys
import time

import lance

sys.path.insert(0, ".")
from common import S3_PROXY, Target  # noqa: E402

COUNTS = os.environ.get("PROXY_COUNT_FILE", "../results/proxy_counts.txt")
LATENCY_MS = float(os.environ.get("PROXY_LATENCY_MS", "20"))
target = Target("s3", sys.argv[1] if len(sys.argv) > 1 else S3_PROXY)


def counts():
    try:
        r, b = open(COUNTS).read().split()
        return int(r), int(b)
    except (FileNotFoundError, ValueError):
        return 0, 0


for layout, col in [("lance_small", "value"), ("lance_tensor", "pixels")]:
    ds = lance.dataset(target.lance_uri(layout), storage_options=target.lance_opts())
    n = ds.count_rows()
    ds.take([0], columns=[col])
    for k in (10, 100):
        rows = list(range(0, n, max(1, n // k)))[:k]
        r0, b0 = counts()
        t0 = time.perf_counter()
        ds.take(rows, columns=[col])
        dt = (time.perf_counter() - t0) * 1000
        r1, b1 = counts()
        print(f"{layout:13} take {k:3} rows: {dt:6.0f} ms, {r1 - r0:4} requests, {(b1 - b0) / 1e6:6.1f} MB "
              f"-> ~{dt / LATENCY_MS:.0f} latency periods for {r1 - r0} requests")

ds = lance.dataset(target.lance_uri("lance_sharded_blob"), storage_options=target.lance_opts())
sizes = ds.to_table(columns=["value"]).column("value").combine_chunks()
sizes = (sizes.storage if hasattr(sizes, "storage") else sizes).field("size").to_pylist()
big = [i for i, s in enumerate(sizes) if s > 1 << 20]
ds.read_blob_ranges("value", [(big[0], 0, 1024)], selector="indices")
for k in (10, 100):
    reqs = [(big[(j * 7) % len(big)], (j * 131_071) % (sizes[big[(j * 7) % len(big)]] - 65536), 65536) for j in range(k)]
    r0, b0 = counts()
    t0 = time.perf_counter()
    ds.read_blob_ranges("value", reqs, selector="indices")
    dt = (time.perf_counter() - t0) * 1000
    r1, b1 = counts()
    print(f"{'blob ranges':13} {k:3} x 64 KB: {dt:6.0f} ms, {r1 - r0:4} requests, {(b1 - b0) / 1e6:6.1f} MB "
          f"-> ~{dt / LATENCY_MS:.0f} latency periods for {r1 - r0} requests")
