# Planetary-Scale Earth Observation Architecture: LanceDB + Zarr + Xarray

This document outlines a unified, dual-table cloud-native storage architecture designed for Earth Observation (EO) data. It merges multi-dimensional array operations (Zarr/Xarray) with high-speed pixel-level vector machine learning indexing (LanceDB via H3/DGGS).

---

## 1. The Architectural Strategy

To solve for both **machine learning workflows** (requiring fast pixel/vector indexing) and **visualization pipelines** (requiring fast 2D rasters), this design implements a **Multi-Table Database Strategy** inside LanceDB. This decouples the Zarr abstraction API from standard file system storage layout complexities.

```
[ USER QUERY LAYER ]
   /             \
  / (ML Search)   \ (Visual / Raster Map)
 ↓                 ↓
[Table 1: H3 Pixels]  ----(Pointer)----> [Table 2: Array Chunks]
(Trillion rows, vectors)              (Packed 2D/3D compressed rasters)
```

---

## 2. Table Schemas & Data Modeling

### Table 1: The Pixel Index Table (`earth_pixels_h3`)
Built for machine learning, vector similarity searches, and discrete point lookups. Every single cell on Earth (at H3 Resolution 11 ~25m) gets its own row containing a 128-dimensional TESSERA embedding alongside precise pointers back to its spatial raster matrix chunk.

*   **h3_id** (String): Unique global planetary hexagon ID.
*   **year** (Int16): Temporal marker.
*   **tessera_embedding** (FixedSizeList[Float32, 128]): Vector payload for semantic similarity indexing.
*   **chunk_id** (String): Pointer to the parent row in Table 2.
*   **pixel_x** (Int16): X-offset array coordinate inside that target chunk.
*   **pixel_y** (Int16): Y-offset array coordinate inside that target chunk.

### Table 2: The Array Chunk Table (`earth_chunks_raster`)
Built for GIS visualization, web map tiling, and bulk regional array loads. Instead of saving thousands of individual chunk files on a file system or object store, data blocks are written as unified binary rows.

*   **chunk_id** (String): Unique relational ID (Primary Key).
*   **grid_system** (String): Grid baseline indicator (e.g., MGRS, UTM).
*   **bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax** (Float64): Spatial boundary parameters.
*   **rgb_nir_data** (Binary): Compressed 4-band x 512 x 512 image block.
*   **tessera_chunk_data** (Binary): Compressed 128-band x 512 x 512 raw embedding matrix.

---

## 3. The Custom Zarr V3 Store Implementation

The custom `LanceDBZarrStore` inherits from Zarr V3's abstract asynchronous storage class. It intercepts standard Xarray chunk array key lookups and maps them directly to high-throughput, index-driven database requests.

```python
import pyarrow as pa
import lancedb
from zarr.abc.store import Store
from typing import AsyncIterator

class LanceDBZarrStore(Store):
    def __init__(self, db_uri: str, chunk_table_name: str = "earth_chunks_raster"):
        super().__init__()
        self.db = lancedb.connect(db_uri)
        self.table = self.db.open_table(chunk_table_name)

    async def get(self, key: str, prototype=None) -> bytes | None:
        # 1. Handle Metadata structural requests
        if ".json" in key or "zarr.json" in key:
            result = self.table.search().where(f"chunk_id == '{key}'").limit(1).to_arrow()
            if len(result) == 0: return None
            return result["tessera_chunk_data"].as_py()

        # 2. Handle Data Chunk matrix requests
        parts = key.split('/')
        if 'c' in parts:
            array_name = parts[0]
            y_chunk, x_chunk = parts[-2], parts[-1]
            
            target_chunk_id = f"chunk_{y_chunk}_{x_chunk}"
            target_column = "tessera_chunk_data" if array_name == "tessera" else "rgb_nir_data"

            result = (self.table.search()
                      .where(f"chunk_id == '{target_chunk_id}'")
                      .select([target_column])
                      .limit(1)
                      .to_arrow())

            if len(result) == 0: return None
            return result[target_column].as_py()
            
        return None

    async def list(self) -> AsyncIterator[str]:
        all_keys = ["zarr.json", "tessera/zarr.json", "rgb/zarr.json"]
        for k in all_keys: yield k
```

---

## 4. End-User Workflow Integration

### Spatial Array Analysis via Xarray
Data scientists can interface with the custom database storage layer using standard, idiomatic multi-dimensional Xarray code:

```python
import xarray as xr

# Initialize store and load metadata lazily
lancedb_store = LanceDBZarrStore(db_uri="s3://my-eo-vault/lancedb_production")
dataset = xr.open_zarr(lancedb_store)

# Slice spatially using physical coordinates
spatial_slice = dataset["rgb"].sel(y=slice(45.0, 45.1), x=slice(-75.0, -74.9))
image_matrix = spatial_slice.values  # Fetches and decompresses via LanceDB on-the-fly
```

### Driving Xarray from ML Vector Searches
By linking the tables via explicit `chunk_id` and pixel coordinate pointers, users can execute semantic queries over a trillion global rows and instantly isolate their corresponding spatial raster windows:

```python
# 1. Find anomaly pixels globally using a 128d IVF_PQ Vector Index
matching_pixels = pixel_table.search(query_vector).limit(5).to_arrow()

winning_chunks = matching_pixels["chunk_id"].to_pylist()
winning_xs = matching_pixels["pixel_x"].to_pylist()
winning_ys = matching_pixels["pixel_y"].to_pylist()

# 2. Load context imagery frames dynamically via the Xarray store
for chunk_id, px, py in zip(winning_chunks, winning_xs, winning_ys):
    visual_context = dataset["rgb"].isel(y=slice(py-50, py+50), x=slice(px-50, px+50)).values
    display_image(visual_context)
```
