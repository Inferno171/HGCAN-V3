"""
audit_node_features.py                                            [HGCAN_V3]
Empirically audit the 33 node-feature columns in the V3 body graphs. A feature
is FLAGGED if it is constant (zero variance -> no signal), almost always zero
(dead), or ever non-finite (NaN/inf -> training hazard). Measures the REAL
tensors, not the spec.

Run on the body cache after a build (or a dry run -- bodies are cached then too):
  python audit_node_features.py --bodies cache_v3/bodies --sample 800
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import torch

# Column layout of the 33-dim node feature (must match step_graph_v3).
SURF = ["plane", "cylinder", "cone", "sphere", "torus", "bezier",
        "bspline", "revolution", "extrusion", "offset", "other", "unknown"]
CURVE = ["line", "circle", "ellipse", "hyperbola", "parabola",
         "bezier", "bspline", "offset", "other"]

COLS = (
    ["type:is_face", "type:is_edge"]
    + [f"face:onehot:{s}" for s in SURF]
    + ["face:log_area", "face:rel_area", "face:holes",
       "face:K_mean", "face:K_std", "face:H_mean", "face:H_std"]
    + [f"edge:onehot:{c}" for c in CURVE]
    + ["edge:log_len", "edge:rel_len", "edge:closed"]
)
assert len(COLS) == 33, len(COLS)

# which columns are populated only on face nodes vs edge nodes
FACE_BLOCK = set(range(2, 2 + 19))          # cols 2..20
EDGE_BLOCK = set(range(2 + 19, 33))         # cols 21..32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bodies", default="cache_v3/bodies")
    ap.add_argument("--sample", type=int, default=800)
    args = ap.parse_args()

    files = sorted(glob.glob(str(Path(args.bodies) / "*.pt")))
    if not files:
        print(f"no .pt under {args.bodies} — build (or dry-run) the cache first.")
        return
    if args.sample:
        files = files[:args.sample]
    print(f"auditing {len(files)} body graphs\n")

    face_rows, edge_rows = [], []
    nonfinite = np.zeros(33, dtype=np.int64)
    for f in files:
        try:
            g = torch.load(f, weights_only=False)
        except Exception:
            continue
        x = g.x.numpy()
        nt = g.node_type.numpy()
        nonfinite += (~np.isfinite(x)).sum(0)
        if (nt == 0).any():
            face_rows.append(x[nt == 0])
        if (nt == 1).any():
            edge_rows.append(x[nt == 1])

    F = np.concatenate(face_rows, 0) if face_rows else np.zeros((0, 33))
    E = np.concatenate(edge_rows, 0) if edge_rows else np.zeros((0, 33))
    ALL = np.concatenate([F, E], 0) if (F.size or E.size) else np.zeros((0, 33))
    print(f"face nodes: {F.shape[0]:,}   edge nodes: {E.shape[0]:,}   "
          f"total: {ALL.shape[0]:,}\n")

    def block_stats(col):
        # evaluate each column only on the nodes where it is meant to be live
        if col in FACE_BLOCK:
            data, where = F[:, col] if F.size else np.array([]), "face"
        elif col in EDGE_BLOCK:
            data, where = E[:, col] if E.size else np.array([]), "edge"
        else:
            data, where = ALL[:, col] if ALL.size else np.array([]), "all"
        return data, where

    print(f"{'#':>2} {'column':28} {'on':5} {'std':>10} {'mean':>10} "
          f"{'%zero':>7} {'flag'}")
    print("-" * 78)
    flagged = []
    for c in range(33):
        data, where = block_stats(c)
        if data.size == 0:
            print(f"{c:>2} {COLS[c]:28} {where:5} {'(no nodes)'}")
            continue
        std = float(np.std(data))
        mean = float(np.mean(data))
        pz = float(np.mean(np.abs(data) < 1e-12)) * 100
        flag = ""
        if nonfinite[c]:
            flag = f"NON-FINITE x{int(nonfinite[c])}"
        elif std < 1e-9:
            flag = "CONSTANT (no signal)"
        elif pz > 99.5:
            flag = "DEAD (>99.5% zero)"
        elif pz > 95:
            flag = "near-dead (>95% zero)"
        if flag:
            flagged.append((c, COLS[c], flag))
        print(f"{c:>2} {COLS[c]:28} {where:5} {std:>10.4f} {mean:>10.4f} "
              f"{pz:>6.1f}% {flag}")

    print("\n" + "=" * 78)
    if not flagged:
        print("VERDICT: all 33 columns carry signal on their block. None unusable.")
    else:
        print("FLAGGED COLUMNS (candidates to drop or investigate):")
        for c, name, why in flagged:
            print(f"  [{c:>2}] {name:28} {why}")
        print("\nNotes:")
        print("  CONSTANT  -> remove; it adds a parameter for zero information.")
        print("  DEAD onehot -> that surface/curve type ~never occurs in B-rep")
        print("     parts (e.g. bezier/offset); harmless but pruneable.")
        print("  NON-FINITE -> must fix in step_graph_v3 before training.")


if __name__ == "__main__":
    main()
