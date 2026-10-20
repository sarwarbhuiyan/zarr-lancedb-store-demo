"""Write the same synthetic cube into every layout and record write time and size."""

import argparse
import os

import numpy as np

from common import LAYOUTS, ROOT, S3_DIRECT, Target, build, dir_size, reset_local, save, synthetic_cube

ap = argparse.ArgumentParser()
ap.add_argument("--target", choices=["local", "s3"], default="local")
ap.add_argument("--layouts", nargs="*", default=list(LAYOUTS))
args = ap.parse_args()

os.makedirs(ROOT, exist_ok=True)
cube = synthetic_cube()
np.save(f"{ROOT}/cube.npy", cube)
print(f"cube {cube.shape} {cube.dtype} raw {cube.nbytes / 1e6:.0f} MB")
target = Target(args.target, S3_DIRECT if args.target == "s3" else None)
rows = []
for layout in args.layouts:
    if target.name == "local":
        reset_local(layout)
    r = build(layout, target, cube)
    if target.name == "local":
        ext = ".zarr" if layout.startswith("zarr_") else ".icechunk" if layout.startswith("icechunk_") else ".lance"
        path = f"{ROOT}/{layout}{ext}"
        r["size_mb"] = round(dir_size(path) / 1e6, 1)
        r["files"] = sum(len(f) for _, _, f in os.walk(path))
    print(r)
    rows.append(r)
save(f"build_{args.target}", rows)
