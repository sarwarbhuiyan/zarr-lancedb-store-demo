"""The read workloads every benchmark runner uses, and the source cube they are checked against."""

import numpy as np

from common import ROOT

rng = np.random.default_rng(42)
AOIS = [(int(rng.integers(0, 924)), int(rng.integers(0, 924))) for _ in range(20)]
PATCHES = [(int(rng.integers(0, 12)), int(rng.integers(0, 992)), int(rng.integers(0, 992))) for _ in range(128)]
cube = np.load(f"{ROOT}/cube.npy", mmap_mode="r")


def aoi(arr, check=False):
    for y, x in AOIS:
        out = arr[:, :, y:y + 100, x:x + 100]
        assert not check or np.array_equal(out, cube[:, :, y:y + 100, x:x + 100])


def patches(arr, check=False):
    for t, y, x in PATCHES:
        out = arr[t, :, y:y + 32, x:x + 32]
        assert not check or np.array_equal(out, cube[t, :, y:y + 32, x:x + 32])


def full_date(arr, check=False):
    out = arr[5]
    assert not check or np.array_equal(out, cube[5])


WORKLOADS = {
    "aoi_20x_100px_all_dates": aoi,
    "patches_128x_32px_one_date": patches,
    "full_date_14x1024x1024": full_date,
}
