"""Vector column type and index type on the dual-table demo's per-pixel embeddings.

Stores the same 262,144 embeddings as float32 and float16, builds IVF_PQ and IVF_RQ on each, and
reports on-disk column and index size, build time, recall@10 against exact cosine search on
float32, and query latency, with default search settings and with every partition probed plus a
re-rank on the stored vectors.
"""

import json
import os
import shutil
import sys
import time

import lancedb
import numpy as np
import pyarrow as pa
from lancedb.index import IvfPq, IvfRq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dual_table import EMB, N, ROOT, pixel_embeddings  # noqa: E402

OUT = os.path.join(ROOT, "vector_compare")
QUERIES, K = 200, 10


def dir_mb(path: str) -> float:
    return round(sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(path) for f in fs) / 1e6, 1)


def main() -> None:
    shutil.rmtree(OUT, ignore_errors=True)
    cube = np.load(f"{ROOT}/cube.npy", mmap_mode="r")[:, :, :N, :N]
    emb = pixel_embeddings(cube, np.random.default_rng(7)).reshape(-1, EMB)
    rng = np.random.default_rng(11)
    qidx = rng.choice(len(emb), QUERIES, replace=False)
    truth = np.argsort(-(emb[qidx] @ emb.T), axis=1)[:, :K]   # exact cosine top-k on float32
    db = lancedb.connect(OUT)
    variants = {"float32": (emb, pa.float32()), "float16": (emb.astype(np.float16), pa.float16())}
    indexes = {
        "IVF_PQ 16x8": IvfPq(distance_type="cosine", num_partitions=64, num_sub_vectors=16),
        "IVF_RQ 1 bit": IvfRq(distance_type="cosine", num_partitions=64, num_bits=1),
        "IVF_RQ 4 bit": IvfRq(distance_type="cosine", num_partitions=64, num_bits=4),
    }
    rows = []
    for vname, (vecs, vtype) in variants.items():
        for iname, cfg in indexes.items():
            name = f"{vname}_{iname.replace(' ', '_')}"
            row = {"vectors": vname, "index": iname}
            col = pa.FixedSizeListArray.from_arrays(pa.array(vecs.reshape(-1), vtype), EMB)
            tbl = db.create_table(name, data=pa.table({"id": pa.array(np.arange(len(vecs))), "vec": col}))
            row["table_mb"] = dir_mb(os.path.join(OUT, f"{name}.lance"))
            t0 = time.perf_counter()
            tbl.create_index("vec", config=cfg)
            row["build_s"] = round(time.perf_counter() - t0, 1)
            row["index_mb"] = round(dir_mb(os.path.join(OUT, f"{name}.lance", "_indices")), 1)
            settings = {"default": {}, "all partitions + refine 10": {"nprobes": 64, "refine_factor": 10}}
            for sname, opts in settings.items():
                hits, times = [], []
                for qi in qidx:
                    q = tbl.search(vecs[qi], vector_column_name="vec").distance_type("cosine").limit(K)
                    if opts:
                        q = q.nprobes(opts["nprobes"]).refine_factor(opts["refine_factor"])
                    t0 = time.perf_counter()
                    got = q.select(["id"]).to_arrow()
                    times.append((time.perf_counter() - t0) * 1000)
                    hits.append(got.column("id").to_numpy())
                recall = float(np.mean([len(set(h) & set(t)) / K for h, t in zip(hits, truth)]))
                out = {**row, "search": sname, "recall@10": round(recall, 3), "query_ms_median": round(float(np.median(times)), 1)}
                rows.append(out)
                print(out, flush=True)
    with open("results/vector_index_compare.json", "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
