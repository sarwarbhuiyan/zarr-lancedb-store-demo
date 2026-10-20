"""Latency of warm REST takes through LanceDB Enterprise: one row, and 64 rows, repeated.

With a warm plan-executor cache the data comes from memory, so whatever remains is the
request path itself (client to query node, the query node's dataset version check, query
node to plan executor). connect_table pins a version; pass --latest to read unpinned, and compare
with the server's --weak-read-consistency-interval-seconds.
"""

import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ENTERPRISE_API_KEY, ENTERPRISE_URL, LANCE_DB  # noqa: E402
from lance_zarr.remote import connect_table, scan_with_row_ids, take_in_order  # noqa: E402

label = sys.argv[1] if len(sys.argv) > 1 else ""
tbl = connect_table("lance_small", database=LANCE_DB, host=ENTERPRISE_URL, api_key=ENTERPRISE_API_KEY)
if "--latest" in sys.argv:
    tbl.checkout_latest()  # unpinned: each request asks the server for the latest version
ids = scan_with_row_ids(tbl, ["key"]).column("_rowid").to_pylist()
for n in (1, 64):
    rows = ids[500:500 + n]
    take_in_order(tbl, rows, "value")  # warm the cache
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        take_in_order(tbl, rows, "value")
        times.append((time.perf_counter() - t0) * 1000)
    print(f"{label:24} warm take of {n:2} rows: median {statistics.median(times):6.1f} ms, min {min(times):6.1f} ms")
