"""
plot_trace.py                                                      [HGCAN_V3]
Turn train_v3's per-epoch trace into figures for the dissertation.

Reads either the results JSON (has a "trace" list) or a *_trace_seed*.csv.
Plots: train loss, val top-1 / top-5, val origin distance — vs epoch.
Overlays multiple seeds if you pass several files.

Run:
  python plot_trace.py reports/v3_results.json
  python plot_trace.py reports/v3_results_trace_seed0.csv reports/..._seed1.csv
  python plot_trace.py reports/v3_results.json --out reports/curves.png
"""

import argparse, csv, json
from pathlib import Path

import matplotlib.pyplot as plt


def load(path):
    p = Path(path)
    if p.suffix == ".json":
        d = json.loads(p.read_text())
        tr = d.get("trace", [])
        label = f"seed {d.get('seed','?')}"
        return label, tr
    # CSV
    rows = []
    with open(p) as f:
        for r in csv.DictReader(f):
            row = {}
            for k, v in r.items():
                if v == "" or v is None:
                    row[k] = None
                elif k == "is_best":
                    row[k] = (v == "True")
                else:
                    row[k] = float(v) if k != "epoch" else int(float(v))
            rows.append(row)
    return p.stem, rows


def series(tr, key):
    xs, ys = [], []
    for r in tr:
        if r.get(key) is not None:
            xs.append(r["epoch"]); ys.append(r[key])
    return xs, ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default="reports/v3_curves.png")
    ap.add_argument("--title", default="HGCAN V3 — training curves")
    args = ap.parse_args()

    runs = [load(f) for f in args.files]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    fig.suptitle(args.title, fontsize=13, y=1.02)

    # 1) loss
    for lab, tr in runs:
        x, y = series(tr, "train_loss")
        axes[0].plot(x, y, label=lab, lw=1.6)
    axes[0].set_title("train loss"); axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("matching CE"); axes[0].grid(alpha=.3)

    # 2) top-1 / top-5
    for lab, tr in runs:
        x1, y1 = series(tr, "val_top1")
        x5, y5 = series(tr, "val_top5")
        l = axes[1].plot(x1, y1, lw=1.8, label=f"{lab} top-1")[0]
        axes[1].plot(x5, y5, lw=1.3, ls="--", color=l.get_color(),
                     label=f"{lab} top-5")
    axes[1].set_title("val entity accuracy"); axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("recall"); axes[1].set_ylim(0, 1); axes[1].grid(alpha=.3)
    axes[1].legend(fontsize=8)

    # 3) origin distance
    for lab, tr in runs:
        x, y = series(tr, "val_org_med")
        axes[2].plot(x, y, label=lab, lw=1.6)
    axes[2].set_title("val origin distance (median)"); axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("mm"); axes[2].grid(alpha=.3)

    for ax in axes:
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=180, bbox_inches="tight")
    print("wrote", args.out)
    try:
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()
