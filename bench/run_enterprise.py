"""Cold, disk-warm and memory-warm reads through LanceDB Enterprise, next to direct Lance reads.

For each Enterprise layout and workload:
  cold    plan executor restarted with its memory and disk cache emptied
  disk    plan executor restarted with only the memory cache emptied (disk tier warm)
  memory  the same reads again with both tiers warm (median of --repeats)
Direct lance_* layouts read MinIO from the client with no cache, so they have one number,
the median of --repeats after one verified pass. Opening a store (key scan, metadata) is
outside every timing. Needs scripts/s3_up.sh and scripts/enterprise.sh running.
"""

import argparse
import os
import statistics
import subprocess
import time

import zarr

from common import S3_PROXY, Target, open_array, save
from workloads import WORKLOADS

ENTERPRISE_SH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "enterprise.sh")

ap = argparse.ArgumentParser()
ap.add_argument("--layouts", nargs="*", default=["ent_small", "ent_tensor", "ent_big", "lance_small", "lance_tensor", "lance_big"])
ap.add_argument("--repeats", type=int, default=2)
ap.add_argument("--concurrency", type=int, default=64, help="zarr async.concurrency")
ap.add_argument("--tag", default="", help="suffix for the results file")
args = ap.parse_args()
zarr.config.set({"async.concurrency": args.concurrency})
target = Target("s3", S3_PROXY)


def restart_pe(cold: bool) -> None:
    subprocess.run(["bash", ENTERPRISE_SH, "restart-pe"] + (["--cold"] if cold else []), check=True,
                   stdout=subprocess.DEVNULL)


def timed(layout: str, fn, check: bool = False) -> float:
    arr, _ = open_array(layout, target)
    t0 = time.perf_counter()
    fn(arr, check=check)
    return round(time.perf_counter() - t0, 3)


results = []
for layout in args.layouts:
    for wname, fn in WORKLOADS.items():
        row = {"layout": layout, "workload": wname}
        if layout.startswith("ent_"):
            restart_pe(cold=True)
            row["cold_s"] = timed(layout, fn, check=True)
            restart_pe(cold=False)
            row["disk_s"] = timed(layout, fn)
            row["memory_s"] = statistics.median(timed(layout, fn) for _ in range(args.repeats))
        else:
            timed(layout, fn, check=True)
            row["direct_s"] = statistics.median(timed(layout, fn) for _ in range(args.repeats))
        print(row, flush=True)
        results.append(row)
save(f"run_enterprise_c{args.concurrency}{args.tag}", results)
