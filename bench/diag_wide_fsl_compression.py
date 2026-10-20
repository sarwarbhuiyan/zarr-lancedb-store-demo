"""Which Lance 2.2 layouts compress image pixels? Wide fixed-size lists (one chip per row) vs others.

Writes the same smooth uint16 chips under each structural encoding with zstd requested and
reports the on-disk size next to the raw size and a whole-array zstd baseline.
"""

import os
import shutil
import sys

import lance
import numpy as np
import pyarrow as pa
from numcodecs import Blosc

OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/lance_fsl_compression"
ROWS, SHAPE = 200, (14, 64, 64)
rng = np.random.default_rng(0)
yy, xx = np.mgrid[0:SHAPE[1], 0:SHAPE[2]]
chips = np.stack([
    (2000 + 800 * np.sin((yy + r) / 9.0)[None] * np.cos((xx - r) / 13.0)[None] + rng.normal(0, 40, SHAPE)).astype(np.uint16)
    for r in range(ROWS)
])
raw = chips.nbytes
width = int(np.prod(SHAPE))


def size_of(path):
    return sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(path) for f in fs if "/data" in d)


def write(name, meta, kind="chip"):
    """Write the same pixels as one chip per row ("chip"), raw chip bytes ("binary"), a narrow
    8-value fixed-size list per row ("fsl8"), or one uint16 per row ("scalar")."""
    flat = pa.array(chips.reshape(-1))
    if kind == "binary":
        field = pa.field("pixels", pa.large_binary(), metadata=meta)
        arr = pa.array([c.tobytes() for c in chips], pa.large_binary())
    elif kind == "scalar":
        field, arr = pa.field("pixels", pa.uint16(), metadata=meta), flat
    else:
        n = width if kind == "chip" else 8
        field = pa.field("pixels", pa.list_(pa.uint16(), n), metadata=meta)
        arr = pa.FixedSizeListArray.from_arrays(flat, n)
    path = os.path.join(OUT, f"{name}.lance")
    shutil.rmtree(path, ignore_errors=True)
    try:
        lance.write_dataset(pa.table([arr], schema=pa.schema([field])), path, data_storage_version="2.2")
        print(f"{name:28} {size_of(path) / raw:5.2f} x raw")
    except Exception as e:  # report the encoder's own error
        print(f"{name:28} error: {str(e).splitlines()[0][:110]}")


print(f"raw {raw / 1e6:.1f} MB, one value = {width * 2} bytes")
blosc = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)
print(f"{'Blosc zstd per chip (Zarr)':28} {sum(len(blosc.encode(c)) for c in chips) / raw:5.2f} x raw")
write("default", {})
write("zstd", {"lance-encoding:compression": "zstd"})
write("fullzip + zstd", {"lance-encoding:compression": "zstd", "lance-encoding:structural-encoding": "fullzip"})
write("miniblock + zstd", {"lance-encoding:compression": "zstd", "lance-encoding:structural-encoding": "miniblock"})
write("binary (raw bytes)", {}, kind="binary")
write("binary + zstd", {"lance-encoding:compression": "zstd"}, kind="binary")
print("same pixels, narrow layouts that fit mini-block:")
write("8-value list, default", {}, kind="fsl8")
write("8-value list + zstd", {"lance-encoding:compression": "zstd"}, kind="fsl8")
write("scalar uint16, default", {}, kind="scalar")
write("scalar uint16 + zstd", {"lance-encoding:compression": "zstd"}, kind="scalar")
