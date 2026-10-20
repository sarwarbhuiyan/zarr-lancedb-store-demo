"""Turn results/*.json into the markdown tables used in the README."""

import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "results"
WL = {"aoi_20x_100px_all_dates": "20 AOIs, 100 px, all dates",
      "patches_128x_32px_one_date": "128 patches, 32 px, one date",
      "full_date_14x1024x1024": "one full date"}


def table(run_file: str) -> str:
    rows = json.loads((R / run_file).read_text())
    layouts = list(dict.fromkeys(r["layout"] for r in rows))
    out = ["| Layout | " + " | ".join(WL.values()) + " |", "|---|" + "---|" * len(WL)]
    for lay in layouts:
        cells = []
        for w in WL:
            r = next(x for x in rows if x["layout"] == lay and x["workload"] == w)
            cells.append(f"{r['median_s']:.2f} s · {r['store_gets']} gets · {r['mb_from_store']:.0f} MB")
        out.append(f"| {lay} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def build(build_file: str) -> str:
    rows = json.loads((R / build_file).read_text())
    out = ["| Layout | Write | Size | Files |", "|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['layout']} | {r['write_s']:.2f} s | {r.get('size_mb', '–')} MB | {r.get('files', '–')} |")
    return "\n".join(out)


if __name__ == "__main__":
    for f in sys.argv[1:]:
        print(f"\n### {f}\n")
        print(build(f) if f.startswith("build_") else table(f))
