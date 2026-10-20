"""Run the read workloads against every layout: timing, store requests and bytes returned."""

import argparse
import statistics
import time

import zarr

from common import LAYOUTS, S3_PROXY, Target, open_array, save
from workloads import WORKLOADS

ap = argparse.ArgumentParser()
ap.add_argument("--target", choices=["local", "s3"], default="local")
ap.add_argument("--layouts", nargs="*", default=list(LAYOUTS))
ap.add_argument("--repeats", type=int, default=3)
ap.add_argument("--concurrency", type=int, default=10, help="zarr async.concurrency")
ap.add_argument("--tag", default="", help="suffix for the results file")
args = ap.parse_args()
zarr.config.set({"async.concurrency": args.concurrency})

target = Target(args.target, S3_PROXY if args.target == "s3" else None)
results = []
for layout in args.layouts:
    for wname, fn in WORKLOADS.items():
        arr, store = open_array(layout, target, counting=True)
        fn(arr, check=True)  # one counted, fully verified pass (also warms caches)
        counts = store.counts
        times = []
        for _ in range(args.repeats):
            arr, _ = open_array(layout, target)
            t0 = time.perf_counter()
            fn(arr)
            times.append(time.perf_counter() - t0)
        row = {
            "layout": layout, "workload": wname, "median_s": round(statistics.median(times), 3),
            "store_gets": counts.gets, "ranged_gets": counts.ranged_gets,
            "mb_from_store": round(counts.bytes / 1e6, 1),
        }
        print(row, flush=True)
        results.append(row)
save(f"run_{args.target}_c{args.concurrency}{args.tag}", results)
