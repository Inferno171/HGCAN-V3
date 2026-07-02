"""
train_v3.py                                                        [HGCAN_V3]
Train HGCANv3 for joint LOCALISATION. Metrics are localisation metrics, NOT
macro-F1: entity top-1/top-5, axis angular error (deg), origin distance (mm).

Loss = multi-positive matching CE  (+ optional 7-class type aux).

Per-assembly processing (batch_size=1 semantics). Assemblies vary hugely in
size and never message-pass across each other, so this is identical in result
to true batching and keeps the multi-positive bookkeeping unambiguous. To speed
up later, flatten joint_pos_* into tensors and register them in __inc__.

Run (from inside HGCAN_V1/, after building cache_v3 with assembly_graph_v3):
  python -m train_v3 --cache-dir cache_v3 --epochs 200 --seed 0
"""

import argparse, json, random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from models.model_v3 import HGCANv3, derive_axis
from data.assembly_graph_v3 import JOINT_TYPES


def seed_everything(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def load_split(cache):
    sp = json.loads((Path(cache) / "splits.json").read_text())
    def load(names):
        return [torch.load(Path(cache) / "assemblies" / f"{n}.pt",
                           weights_only=False) for n in names]
    return sp, load(sp["train"]), load(sp["test"])


# ----------------------------------------------------------------- loss
def matching_loss(mats, idx_i, idx_j, data, type_logits=None, lam_type=0.3):
    loss = mats[0].new_zeros(())
    n = 0
    for p, mat in enumerate(mats):
        Ei, Ej = mat.shape
        if Ei == 0 or Ej == 0:
            continue
        logp = F.log_softmax(mat.reshape(-1), dim=0)
        a = torch.isin(idx_i[p], data.joint_pos_i[p].to(idx_i[p].device))
        b = torch.isin(idx_j[p], data.joint_pos_j[p].to(idx_j[p].device))
        cell = (a.unsqueeze(1) & b.unsqueeze(0)).reshape(-1)
        if cell.any():
            loss = loss - torch.logsumexp(logp[cell], dim=0)
            n += 1
    loss = loss / max(n, 1)
    if type_logits is not None and type_logits.size(0):
        loss = loss + lam_type * F.cross_entropy(type_logits, data.joint_type)
    return loss


# ----------------------------------------------------------------- metrics
@torch.no_grad()
def evaluate(model, ds, device):
    model.eval()
    top1 = top5 = tot = 0
    ang, org = [], []
    type_pred, type_true = [], []          # 7-class type head (if present)
    for data in ds:
        data = data.to(device)
        z_ent, h_occ_ctx = model.encode(data)
        mats, idx_i, idx_j = model.match_pairs(z_ent, data)
        # ---- auxiliary joint-type prediction (only if the head exists) ----
        if model.type_head is not None and hasattr(data, "joint_type"):
            tl = model.type_head(h_occ_ctx, data.joint_occ_pairs)
            if tl.size(0):
                type_pred.append(tl.argmax(-1).cpu())
                type_true.append(data.joint_type.cpu())
        for p, mat in enumerate(mats):
            Ei, Ej = mat.shape
            if Ei == 0 or Ej == 0:
                continue
            tot += 1
            flat = mat.reshape(-1)
            order = torch.argsort(flat, descending=True)
            pos_i = data.joint_pos_i[p].to(device)
            pos_j = data.joint_pos_j[p].to(device)

            def hit(k):
                for c in order[:k]:
                    ai, bj = int(c) // Ej, int(c) % Ej
                    if (idx_i[p][ai] in pos_i) and (idx_j[p][bj] in pos_j):
                        return True
                return False
            top1 += int(hit(1)); top5 += int(hit(5))

            # axis metrics from the top-1 predicted entity on side i
            c0 = int(order[0]); ai = c0 // Ej
            pred_ent = int(idx_i[p][ai])
            ax = derive_axis(data.entity_axis, pred_ent)
            # GT axis is read from the TRUE entity's own analytic geometry, so
            # prediction and GT share the occurrence-local frame (no world/local
            # mismatch). Pick the first positive entity on side i with a valid axis.
            gt = None
            for te in pos_i.tolist():
                cand = derive_axis(data.entity_axis, int(te))
                if cand is not None:
                    gt = cand; break
            if ax is not None and gt is not None:
                pd = ax[3:] / (torch.linalg.norm(ax[3:]) + 1e-9)
                gd = gt[3:] / (torch.linalg.norm(gt[3:]) + 1e-9)
                cos = float(torch.clamp(torch.abs((pd * gd).sum()), max=1.0))
                ang.append(np.degrees(np.arccos(cos)))
                # origin distance: true-entity origin to predicted entity's axis line
                w = (gt[:3] - ax[:3])
                org.append(float(torch.linalg.norm(
                    w - (w * pd).sum() * pd)))
    def dist_stats(vals, prefix):
        if not vals:
            return {f"{prefix}_med": float("nan"), f"{prefix}_mean": float("nan"),
                    f"{prefix}_p90": float("nan"), f"{prefix}_zero_frac": float("nan"),
                    f"{prefix}_n": 0}
        a = np.asarray(vals, dtype=float)
        return {f"{prefix}_med": float(np.median(a)),
                f"{prefix}_mean": float(a.mean()),
                f"{prefix}_p90": float(np.percentile(a, 90)),
                f"{prefix}_zero_frac": float((a < 0.1).mean()),
                f"{prefix}_n": int(a.size)}
    res = dict(top1=top1 / max(tot, 1), top5=top5 / max(tot, 1), n=tot)
    res.update(dist_stats(ang, "ang"))
    res.update(dist_stats(org, "org"))
    # ---- joint-type metrics: accuracy + macro-F1 over support>=5 classes ----
    # (matches the V2 convention so the number is comparable to the 0.253 baseline)
    if type_pred:
        yp = torch.cat(type_pred); yt = torch.cat(type_true)
        n_cls = len(JOINT_TYPES)
        cm = torch.zeros(n_cls, n_cls, dtype=torch.long)
        for t, p in zip(yt.tolist(), yp.tolist()):
            cm[t, p] += 1
        support = cm.sum(1)
        tp = cm.diag().float()
        prec = tp / cm.sum(0).clamp_min(1)
        rec = tp / cm.sum(1).clamp_min(1)
        f1 = 2 * prec * rec / (prec + rec).clamp_min(1e-12)
        present = support >= 5
        if not present.any():
            present = support > 0
        res["type_acc"] = float((yp == yt).float().mean())
        res["type_macro_f1"] = float(f1[present].mean())
        res["type_n"] = int(yt.numel())
    else:
        res["type_acc"] = float("nan")
        res["type_macro_f1"] = float("nan")
        res["type_n"] = 0
    return res


# ----------------------------------------------------------------- train
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="cache_v3")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--geo-dim", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lam-type", type=float, default=0.3,
                    help="weight on the auxiliary 7-class type loss (0 disables)")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overfit", type=int, default=0,
                    help="overfit test: train+val on the first N assemblies only")
    ap.add_argument("--out", default="reports/v3_results.json")
    args = ap.parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    _, train_all, test_ds = load_split(args.cache_dir)
    rng = random.Random(args.seed); rng.shuffle(train_all)
    if args.overfit:
        train_ds = val_ds = train_all[:args.overfit]     # memorise a tiny subset
        print(f"OVERFIT TEST: {len(train_ds)} assemblies (train == val)")
    else:
        n_val = max(2, int(args.val_frac * len(train_all)))
        val_ds, train_ds = train_all[:n_val], train_all[n_val:]
    print(f"assemblies: {len(train_ds)} train / {len(val_ds)} val / "
          f"{len(test_ds)} test")

    model = HGCANv3(geo_dim=args.geo_dim, dropout=args.dropout,
                    with_type_head=args.lam_type > 0).to(device)
    print("params:", sum(p.numel() for p in model.parameters()))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best, best_state, best_ep = -1.0, None, -1
    trace = []                                   # per-epoch log for plotting
    for ep in range(1, args.epochs + 1):
        model.train()
        rng.shuffle(train_ds)
        tot = 0.0
        for data in train_ds:
            data = data.to(device)
            mats, idx_i, idx_j, type_logits = model(data)
            loss = matching_loss(mats, idx_i, idx_j, data, type_logits,
                                 args.lam_type)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach())
        train_loss = tot / max(len(train_ds), 1)
        row = {"epoch": ep, "train_loss": round(train_loss, 5)}
        if ep % 2 == 0 or ep == 1:
            vm = evaluate(model, val_ds, device)
            flag = ""
            if vm["top1"] > best:
                best, best_ep = vm["top1"], ep
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
                flag = "  *best*"
            row.update(val_top1=round(vm["top1"], 4),
                       val_top5=round(vm["top5"], 4),
                       val_ang_med=None if vm["ang_med"] != vm["ang_med"] else round(vm["ang_med"], 3),
                       val_ang_mean=None if vm["ang_mean"] != vm["ang_mean"] else round(vm["ang_mean"], 3),
                       val_org_med=None if vm["org_med"] != vm["org_med"] else round(vm["org_med"], 3),
                       val_org_mean=None if vm["org_mean"] != vm["org_mean"] else round(vm["org_mean"], 3),
                       val_org_p90=None if vm["org_p90"] != vm["org_p90"] else round(vm["org_p90"], 3),
                       val_org_zero_frac=None if vm["org_zero_frac"] != vm["org_zero_frac"] else round(vm["org_zero_frac"], 3),
                       val_type_acc=None if vm["type_acc"] != vm["type_acc"] else round(vm["type_acc"], 4),
                       val_type_macro_f1=None if vm["type_macro_f1"] != vm["type_macro_f1"] else round(vm["type_macro_f1"], 4),
                       is_best=bool(flag))
            tstr = ("" if vm["type_n"] == 0
                    else f"  type acc {vm['type_acc']:.3f} mF1 {vm['type_macro_f1']:.3f}")
            print(f"ep {ep:>3}  loss {train_loss:.4f}  "
                  f"val top1 {vm['top1']:.3f} top5 {vm['top5']:.3f}  "
                  f"ang(med/mean) {vm['ang_med']:.1f}/{vm['ang_mean']:.1f}  "
                  f"org(med/mean/p90) {vm['org_med']:.1f}/{vm['org_mean']:.1f}/{vm['org_p90']:.1f}mm  "
                  f"[org0 {vm['org_zero_frac']*100:.0f}% n{vm['org_n']}]{tstr}{flag}")
            if ep - best_ep > args.patience:
                trace.append(row)
                print("early stop"); break
        trace.append(row)

    model.load_state_dict(best_state); model.to(device)
    tm = evaluate(model, test_ds, device)
    print("\n--- TEST (localisation) ---")
    print(f"entity top-1 {tm['top1']:.3f}  top-5 {tm['top5']:.3f}  "
          f"(n={tm['n']})")
    print(f"axis angular error   median {tm['ang_med']:.2f}  mean {tm['ang_mean']:.2f} deg")
    print(f"origin distance      median {tm['org_med']:.2f}  mean {tm['org_mean']:.2f}  "
          f"p90 {tm['org_p90']:.2f} mm")
    print(f"  (axis exact on {tm['org_zero_frac']*100:.0f}% of {tm['org_n']} valid-axis joints "
          f"-> axis recovered even when exact entity isn't)")
    if tm["type_n"]:
        print(f"joint type (aux)     acc {tm['type_acc']:.3f}  "
              f"macro-F1 {tm['type_macro_f1']:.3f}  (support>=5 classes, V2-comparable)")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"seed": args.seed, "best_epoch": best_ep,
                               "test": tm, "config": vars(args),
                               "trace": trace}, indent=1))
    torch.save({"model": best_state, "config": vars(args)},
               out.with_suffix(".ckpt"))
    # trace as CSV too — one row per epoch, ready to plot
    trace_csv = out.with_name(out.stem + f"_trace_seed{args.seed}.csv")
    cols = ["epoch", "train_loss", "val_top1", "val_top5",
            "val_ang_med", "val_ang_mean",
            "val_org_med", "val_org_mean", "val_org_p90", "val_org_zero_frac",
            "val_type_acc", "val_type_macro_f1",
            "is_best"]
    lines = [",".join(cols)]
    for r in trace:
        lines.append(",".join("" if r.get(c) is None else str(r.get(c, ""))
                              for c in cols))
    trace_csv.write_text("\n".join(lines))
    print("wrote", out, "and", trace_csv)


if __name__ == "__main__":
    main()
