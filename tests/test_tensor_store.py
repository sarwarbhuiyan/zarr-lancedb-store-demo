"""LanceTensorStore serves correct windows through zarr and xarray, and the same rows answer SQL."""

import lance
import numpy as np
import pyarrow as pa
import xarray as xr
import zarr

from lance_zarr import LanceTensorStore


def _chip_table(cube: np.ndarray, chip: int) -> pa.Table:
    t_n, b_n, h, w = cube.shape
    rows = [(t, i, j) for t in range(t_n) for i in range(h // chip) for j in range(w // chip)]
    chips = np.stack([cube[t, :, i * chip:(i + 1) * chip, j * chip:(j + 1) * chip] for t, i, j in rows])
    return pa.table({
        "t": pa.array([r[0] for r in rows], pa.int32()),
        "ty": pa.array([r[1] for r in rows], pa.int32()),
        "tx": pa.array([r[2] for r in rows], pa.int32()),
        "cloud_frac": pa.array(np.linspace(0, 1, len(rows)), pa.float32()),
        "pixels": pa.FixedShapeTensorArray.from_numpy_ndarray(chips),
    })


def test_windows_and_sql(tmp_path):
    rng = np.random.default_rng(0)
    cube = rng.integers(0, 10000, size=(3, 4, 64, 96), dtype=np.uint16)
    uri = str(tmp_path / "chips.lance")
    lance.write_dataset(_chip_table(cube, 16), uri)
    store = LanceTensorStore(uri)
    arr = zarr.open_array(store, path="reflectance", mode="r")
    assert arr.shape == cube.shape and arr.chunks == (1, 4, 16, 16)
    assert np.array_equal(arr[:, :, 5:37, 11:70], cube[:, :, 5:37, 11:70])   # window across chips
    ds = xr.open_zarr(store, consolidated=False, zarr_format=3)
    assert np.array_equal(ds["reflectance"].isel(time=1, y=slice(0, 20)).values, cube[1, :, 0:20])
    clear = lance.dataset(uri).to_table(filter="cloud_frac < 0.1", columns=["t", "ty", "tx"])
    assert clear.num_rows > 0
