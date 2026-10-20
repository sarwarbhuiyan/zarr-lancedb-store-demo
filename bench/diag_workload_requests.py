"""HTTP requests and bytes each layout sends for each workload, counted by a proxy.

Run scripts/latency_proxy.py in front of the S3 server and pass its URL. Opening the array
(manifests, key maps, blob handles, zarr.json) happens before counting starts, so the counts
are for the reads alone.
"""

import os
import sys
import time

import numpy as np
import zarr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Target, open_array  # noqa: E402

COUNTS = os.environ.get("PROXY_COUNT_FILE", "../results/proxy_counts_minio.txt")
target = Target("s3", sys.argv[1])
zarr.config.set({"async.concurrency": int(os.environ.get("CONCURRENCY", "10"))})
rng = np.random.default_rng(42)
AOIS = [(int(rng.integers(0, 924)), int(rng.integers(0, 924))) for _ in range(20)]
PATCHES = [(int(rng.integers(0, 12)), int(rng.integers(0, 992)), int(rng.integers(0, 992))) for _ in range(128)]
WORKLOADS = {
    "20 AOIs": lambda a: [a[:, :, y:y + 100, x:x + 100] for y, x in AOIS],
    "128 patches": lambda a: [a[t, :, y:y + 32, x:x + 32] for t, y, x in PATCHES],
    "one full date": lambda a: a[5],
}


def counts():
    try:
        r, b = open(COUNTS).read().split()
        return int(r), int(b)
    except (FileNotFoundError, ValueError):
        return 0, 0


print(f"{'layout':20} {'workload':14} {'requests':>8} {'MB':>7} {'seconds':>8}")
for layout in sys.argv[2:]:
    for name, fn in WORKLOADS.items():
        arr, _ = open_array(layout, target)
        r0, b0 = counts()
        t0 = time.perf_counter()
        fn(arr)
        dt = time.perf_counter() - t0
        time.sleep(0.3)  # let the proxy flush its count file
        r1, b1 = counts()
        print(f"{layout:20} {name:14} {r1 - r0:8} {(b1 - b0) / 1e6:7.1f} {dt:8.2f}", flush=True)
