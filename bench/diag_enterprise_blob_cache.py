"""Does LanceDB Enterprise cache Blob V2 reads? Object-store requests and time per pass.

For each layout and workload: restart the plan executor with empty caches, open the store, then
read the workload twice through the REST API. A counting proxy sits between both Enterprise
processes and MinIO (start Enterprise with S3_ENDPOINT pointing at it), so the counts are every
request that reaches object storage. A cached path sends close to nothing on the second pass.
Opening a store (key scan, blob handles) happens before counting starts.

Usage: PROXY_COUNT_FILE=... uv run python diag_enterprise_blob_cache.py ent_small ent_sharded_blob ...
"""

import os
import subprocess
import sys
import time

import zarr

from common import S3_PROXY, Target, open_array
from workloads import WORKLOADS

ENTERPRISE_SH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "enterprise.sh")
COUNTS = os.environ["PROXY_COUNT_FILE"]
PASSES = ("cold", "warm")
SELECTED = {
    "128 patches": WORKLOADS["patches_128x_32px_one_date"],
    "one full date": WORKLOADS["full_date_14x1024x1024"],
}


def counts() -> tuple[int, int]:
    try:
        r, b = open(COUNTS).read().split()
        return int(r), int(b)
    except (FileNotFoundError, ValueError):
        return 0, 0


def settled_counts() -> tuple[int, int]:
    time.sleep(0.5)  # let the proxy flush its count file
    return counts()


zarr.config.set({"async.concurrency": 64})
target = Target("s3", S3_PROXY)
print(f"{'layout':18} {'workload':14} {'pass':5} {'requests':>8} {'MB':>7} {'seconds':>8}", flush=True)
for layout in sys.argv[1:]:
    for name, fn in SELECTED.items():
        subprocess.run(["bash", ENTERPRISE_SH, "restart-pe", "--cold"], check=True, stdout=subprocess.DEVNULL)
        arr, _ = open_array(layout, target)
        for p in PASSES:
            r0, b0 = settled_counts()
            t0 = time.perf_counter()
            fn(arr, check=p == "cold")
            dt = time.perf_counter() - t0
            r1, b1 = settled_counts()
            print(f"{layout:18} {name:14} {p:5} {r1 - r0:8} {(b1 - b0) / 1e6:7.1f} {dt:8.2f}", flush=True)
