"""Object-store requests behind a cold and a warm pass through Enterprise, next to a direct read.

Run with scripts/latency_proxy.py on :19002 in front of MinIO, and scripts/enterprise.sh started
with S3_ENDPOINT=http://127.0.0.1:19002 so the query node and plan executor read through it.
Counts are taken after the store opens, so they cover the reads alone.
"""

import os
import subprocess
import sys
import time

import zarr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Target, open_array  # noqa: E402
from workloads import WORKLOADS  # noqa: E402

COUNTS = os.environ.get("PROXY_COUNT_FILE", "../results/proxy_counts_minio.txt")
ENTERPRISE_SH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "enterprise.sh")
zarr.config.set({"async.concurrency": 64})
proxy = Target("s3", "http://localhost:19002")


def counts():
    try:
        r, b = open(COUNTS).read().split()
        return int(r), int(b)
    except (FileNotFoundError, ValueError):
        return 0, 0


def measure(label, layout, fn):
    arr, _ = open_array(layout, proxy)
    time.sleep(0.3)
    r0, b0 = counts()
    t0 = time.perf_counter()
    fn(arr)
    dt = time.perf_counter() - t0
    time.sleep(0.3)
    r1, b1 = counts()
    print(f"{label:28} {r1 - r0:6} requests {(b1 - b0) / 1e6:7.1f} MB {dt:7.2f} s", flush=True)


for wname in sys.argv[1:] or ["aoi_20x_100px_all_dates"]:
    fn = WORKLOADS[wname]
    print(f"-- {wname}")
    for layout in ("ent_small", "ent_tensor"):
        subprocess.run(["bash", ENTERPRISE_SH, "restart-pe", "--cold"], check=True, stdout=subprocess.DEVNULL)
        measure(f"{layout} cold", layout, fn)
        measure(f"{layout} warm", layout, fn)
    for layout in ("lance_small", "lance_tensor"):
        measure(f"{layout} direct", layout, fn)
