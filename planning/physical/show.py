"""Print scoreboard tables as Markdown.

    python show.py seeds [--it 135168] [--n 1e6] [--mc 0.1,0.2] [--match substr]
    python show.py single [--it 135168] [--match substr] [--mc ...]
"""

import argparse
import csv

from common import RES


def read(name):
    with open(RES / "tables" / name) as fh:
        return list(csv.DictReader(fh))


def f(x, nd=3):
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["seeds", "single"])
    ap.add_argument("--it", default="135168")
    ap.add_argument("--n", default="")
    ap.add_argument("--mc", default="")
    ap.add_argument("--match", default="")
    ap.add_argument("--sort", default="")
    a = ap.parse_args()
    mcs = {float(m) for m in a.mc.split(",") if m}
    if a.kind == "seeds":
        rows = [r for r in read("seeds.csv") if r["it"] == a.it]
        cols = ["config", "mc", "seeds", "K", "C", "D1", "D3", "D", "S_seed", "P_seed", "P_seed_std", "P_T1",
                "P_T3", "sheet_vf", "sheet_prec", "sheet_rec"]
    else:
        rows = [r for r in read("scoreboard.csv") if r["it"] == a.it]
        cols = ["tag", "mc", "K", "C", "D1", "D3", "D", "S_mc", "P_mc", "sheet_vf", "sheet_prec", "sheet_rec",
                "sheet_prec3", "sheet_rec3", "sheet_kappa", "sheet_ridge_enr", "sheet_chi_med", "speckle_vw",
                "sheet_speckle"]
    rows = [r for r in rows if (not mcs or float(r["mc"]) in mcs) and a.match in r.get("config", r.get("tag"))]
    if a.n:
        rows = [r for r in rows if f"-N{a.n}-" in r.get("config", r.get("tag"))]
    key = (lambda r: -float(r[a.sort] or 0)) if a.sort else (lambda r: (r.get("config", r.get("tag")), float(r["mc"])))
    rows.sort(key=key)
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows:
        out = []
        for c in cols:
            v = r.get(c, "")
            if c in ("config", "tag"):
                out.append(v.replace("-it180224", "").replace("-a0.5-s1", ""))
            elif c in ("mc", "seeds"):
                out.append(v)
            elif c == "K":
                out.append(f(v, 1))
            else:
                out.append(f(v))
        print("| " + " | ".join(out) + " |")


if __name__ == "__main__":
    main()
