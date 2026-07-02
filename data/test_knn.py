"""
test_knn.py                                                       [HGCAN_V3]
Does the kNN occurrence graph actually connect the parts that are JOINTED?

For every jointed occurrence pair (the supervision), check whether the two
occurrences are within each other's k nearest neighbours by center-of-mass.
High recall  -> kNN structure tracks the joints (useful prior).
Low recall   -> kNN mostly wires unrelated neighbours (connectivity only).

Also reports the same for contacts, and a k-sweep so you can see whether k=8
is the right cut. Reads the built AssemblyDataV3 cache (no recompute).

Run:
  python test_knn.py --cache cache_v3
  python test_knn.py --cache cache_v3 --ks 4 8 12 16 24
"""

import argparse
import glob
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from data.assembly_graph_v3 import REL_CONTACT, REL_KNN


def neighbours_within_k(pos, k):
    """pos (N,3) -> set of undirected pairs (i<j) that are mutually OR singly
    within k nearest. We use 'either direction' (i in kNN(j) OR j in kNN(i)),
    which matches how the graph adds edges."""
    N = pos.shape[0]
    if N <= 1:
        return set()
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    kk = min(k, N - 1)
    nbr = np.argpartition(d, kk - 1, axis=1)[:, :kk]
    pairs = set()
    for i in range(N):
        for j in nbr[i]:
            pairs.add((min(i, int(j)), max(i, int(j))))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache_v3")
    ap.add_argument("--ks", type=int, nargs="+", default=[4, 8, 12, 16, 24])
    args = ap.parse_args()

    files = sorted(glob.glob(str(Path(args.cache) / "assemblies" / "*.pt")))
    if not files:
        print(f"no assemblies under {args.cache}/assemblies — build the cache first.")
        return
    print(f"testing kNN over {len(files)} assemblies\n")

    # we need per-occurrence positions. The cache stores asm_edge_index/type but
    # not pos directly, so we reconstruct kNN-membership from the stored kNN edges
    # for k=8 (the build's k), and recompute for other k from... we DON'T have pos.
    # -> So we report: (a) recall using the STORED graph edges (contact / knn as
    #    built), and (b) note that the k-sweep needs pos, handled below if present.

    has_pos = False
    total_joints = 0
    in_knn = in_contact = in_either = 0
    n_occ_hist = Counter()
    knn_sweep = {k: 0 for k in args.ks}
    sweep_joints = 0

    for f in files:
        d = torch.load(f, weights_only=False)
        P = int(d.num_occ)
        jp = d.joint_occ_pairs
        if jp.numel() == 0:
            continue
        joint_pairs = {(min(int(a), int(b)), max(int(a), int(b)))
                       for a, b in jp.t().tolist()}
        total_joints += len(joint_pairs)
        n_occ_hist[P] += 1

        # stored-graph membership (as actually built, k=8)
        ei, et = d.asm_edge_index, d.asm_edge_type
        knn_e = set(); con_e = set()
        for c in range(ei.size(1)):
            a, b = int(ei[0, c]), int(ei[1, c])
            key = (min(a, b), max(a, b))
            r = int(et[c])
            if r == REL_KNN: knn_e.add(key)
            elif r == REL_CONTACT: con_e.add(key)
        for jpair in joint_pairs:
            ik = jpair in knn_e
            ic = jpair in con_e
            in_knn += ik; in_contact += ic; in_either += (ik or ic)

        # k-sweep needs positions; use them if the cache carries pos
        pos = getattr(d, "pos", None)
        if pos is not None:
            has_pos = True
            pos = pos.numpy()
            sweep_joints += len(joint_pairs)
            for k in args.ks:
                kp = neighbours_within_k(pos, k)
                knn_sweep[k] += sum(1 for jpair in joint_pairs if jpair in kp)

    print("=" * 60)
    print(f"jointed pairs analysed : {total_joints}")
    print(f"recall in kNN edges    : {in_knn}/{total_joints} "
          f"= {100*in_knn/max(total_joints,1):.1f}%")
    print(f"recall in CONTACT edges: {in_contact}/{total_joints} "
          f"= {100*in_contact/max(total_joints,1):.1f}%")
    print(f"recall in EITHER       : {in_either}/{total_joints} "
          f"= {100*in_either/max(total_joints,1):.1f}%  <- candidate coverage")
    print()
    print("occurrence-count distribution (assemblies):")
    for n, c in sorted(n_occ_hist.items())[:12]:
        print(f"  {n:3} occ : {c}")

    if has_pos:
        print("\nk-sweep (recall of jointed pairs within k nearest, from pos):")
        for k in args.ks:
            print(f"  k={k:2} : {100*knn_sweep[k]/max(sweep_joints,1):.1f}%")
    else:
        print("\n(k-sweep skipped: cache has no 'pos' field; recall above uses "
              "the built k=8 graph. To sweep k, add pos to AssemblyDataV3.)")

    print("\nINTERPRETATION")
    print("  high kNN recall  -> kNN connects jointed parts: useful prior.")
    print("  low kNN recall   -> kNN mostly connectivity; signal is weak.")
    print("  EITHER >> kNN    -> contacts carry the joints kNN misses.")


if __name__ == "__main__":
    main()
