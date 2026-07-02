"""
train_v3_classify.py                                              [HGCAN_V3]
Dedicated 7-class JOINT-TYPE classifier on the V3 (edge+face) cache.

Why this file exists
--------------------
V3's main run only has a *weak auxiliary* type head (lam_type=0.3, sharing an
encoder optimised for localisation). That is not a fair test of whether the
richer edge+face representation helps type classification. This script trains
type as the SOLE objective, with proper class balancing, on the V3 cache, and
reports macro-F1 over support>=5 classes + the full confusion matrix -- the
same convention as the V2 baseline (macro-F1 0.253), so the two are directly
comparable.

It reuses V3's own modules (no V2 files, which cannot read the V3 cache):
    EntityEncoder  -> AssemblyContext  -> TypeHead   (all from model_v3.py)

Clean-comparison rules baked in:
  * same official split (splits.json), test touched once at the end
  * distribution-matched val carved from train (no test leakage)
  * macro-F1 over support>=5 classes (V2 convention) + accuracy
  * NO leakage: inputs are geometry only (x_ent, graph edges); joint_type is
    the label, never an input.

Run (Kaggle, from the repo root):
  python -m train_v3_classify --cache-dir "$CACHE" --epochs 80 --seed 0 \
      --out /kaggle/working/reports/v3_cls_seed0.json
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.assembly_graph_v3 import JOINT_TYPES
from models.model_v3 import EntityEncoder, AssemblyContext, TypeHead

NUM_CLASSES = len(JOINT_TYPES)


# ----------------------------------------------------------------- utils
def seed_everything(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def load_split(cache):
    splits = json.loads((cache / "splits.json").read_text())
    def load(names):
        return [torch.load(cache / "assemblies" / f"{n}.pt",
                           weights_only=False) for n in names]
    return splits, load(splits["train"]), load(splits["test"])


def focal_loss(logits, target, alpha, gamma=2.0):
    logp = F.log_softmax(logits, dim=-1)
    logp_t = logp.gather(1, target.unsqueeze(1)).squeeze(1)
    p_t = logp_t.exp()
    return (-alpha[target] * (1 - p_t).pow(gamma) * logp_t).mean()


# ----------------------------------------------------------------- model
class TypeClassifier(nn.Module):
    """EntityEncoder -> AssemblyContext -> TypeHead. Type-only; no matching."""

    def __init__(self, geo_dim=256, dropout=0.1):
        super().__init__()
        self.encoder = EntityEncoder(out_dim=geo_dim, hidden=128, dropout=dropout)
        self.context = AssemblyContext(dim=geo_dim, dropout=dropout)
        self.no_geo = nn.Parameter(torch.zeros(geo_dim))
        self.head = TypeHead(dim=geo_dim, dropout=dropout)

    def forward(self, batch):
        N = int(batch.num_occ.sum()) if torch.is_tensor(batch.num_occ) else batch.num_occ
        h_ent, h_occ = self.encoder(batch.x_ent, batch.ent_edge_index,
                                    batch.ent_edge_type, batch.ent_to_occ, N)
        has_geo = batch.occ_has_geom
        h_occ = torch.where(has_geo.unsqueeze(-1), h_occ,
                            self.no_geo.unsqueeze(0).expand_as(h_occ))
        h_ctx = self.context(h_occ, batch.asm_edge_index, batch.asm_edge_type)
        return self.head(h_ctx, batch.joint_occ_pairs)


# ----------------------------------------------------------------- eval
@torch.no_grad()
def evaluate(model, ds, device, min_support=5):
    model.eval()
    ys, ps = [], []
    for data in ds:
        data = data.to(device)
        logits = model(data)
        if logits.size(0) == 0:
            continue
        ys.append(data.joint_type.cpu())
        ps.append(logits.argmax(-1).cpu())
    y, p = torch.cat(ys), torch.cat(ps)
    cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    for yi, pi in zip(y.tolist(), p.tolist()):
        cm[yi, pi] += 1
    support = cm.sum(1); tp = cm.diag().float()
    prec = tp / cm.sum(0).clamp_min(1); rec = tp / cm.sum(1).clamp_min(1)
    f1 = 2 * prec * rec / (prec + rec).clamp_min(1e-12)
    present = support >= min_support
    if not present.any():
        present = support > 0
    return dict(acc=(y == p).float().mean().item(),
                macro_f1=f1[present].mean().item(),
                cm=cm, f1=f1, prec=prec, rec=rec, support=support)


def print_eval(tag, m):
    print(f"\n--- {tag} ---")
    print(f"accuracy {m['acc']:.4f}   macro-F1 (support>=5) {m['macro_f1']:.4f}")
    print(f"{'class':<24}{'prec':>7}{'rec':>7}{'f1':>7}{'support':>9}")
    for i, t in enumerate(JOINT_TYPES):
        print(f"{t:<24}{m['prec'][i]:>7.3f}{m['rec'][i]:>7.3f}"
              f"{m['f1'][i]:>7.3f}{int(m['support'][i]):>9}")
    print("confusion matrix (rows=true, cols=pred):")
    hdr = " " * 24 + "".join(f"{t[:4]:>6}" for t in JOINT_TYPES)
    print(hdr)
    for i, t in enumerate(JOINT_TYPES):
        print(f"{t:<24}" + "".join(f"{v:>6}" for v in m["cm"][i].tolist()))


# ----------------------------------------------------------------- train
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="cache_v3")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--geo-dim", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--alpha-mode", choices=["inv", "inv_sqrt", "uniform"],
                    default="inv",
                    help="class weighting; 'inv' = full inverse-freq (strong, "
                         "recommended to fight the all-Rigid collapse)")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="reports/v3_cls_seed0.json")
    args = ap.parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    cache = Path(args.cache_dir)
    splits, train_all, test_ds = load_split(cache)

    # distribution-matched val split (by assembly; test untouched)
    def label_vec(ds):
        v = torch.zeros(NUM_CLASSES)
        for d in ds:
            for c in d.joint_type.tolist():
                v[c] += 1
        return v
    gdist = label_vec(train_all); gdist = gdist / gdist.sum().clamp_min(1)
    n_val = max(2, int(args.val_frac * len(train_all)))
    pool = list(range(len(train_all))); random.Random(args.seed).shuffle(pool)
    val_idx, vc = [], torch.zeros(NUM_CLASSES)
    for _ in range(n_val):
        best_i, best_err = None, None
        for i in pool:
            cand = vc + label_vec([train_all[i]])
            err = ((cand / cand.sum().clamp_min(1)) - gdist).abs().sum().item()
            if best_err is None or err < best_err:
                best_i, best_err = i, err
        val_idx.append(best_i); vc += label_vec([train_all[best_i]])
        pool.remove(best_i)
    val_ds = [train_all[i] for i in val_idx]
    train_ds = [train_all[i] for i in pool]
    print(f"assemblies: {len(train_ds)} train / {len(val_ds)} val / "
          f"{len(test_ds)} test")

    # class weights from train
    counts = Counter()
    for d in train_ds:
        for c in d.joint_type.tolist():
            counts[c] += 1
    freq = torch.tensor([counts.get(i, 0) for i in range(NUM_CLASSES)],
                        dtype=torch.float)
    print("train type counts:", freq.int().tolist(),
          "->", dict(zip([t[:4] for t in JOINT_TYPES], freq.int().tolist())))
    if args.alpha_mode == "uniform":
        alpha = torch.ones(NUM_CLASSES)
    elif args.alpha_mode == "inv_sqrt":
        alpha = 1.0 / freq.clamp_min(1.0).sqrt()
    else:                                       # inv (strong)
        alpha = 1.0 / freq.clamp_min(1.0)
    alpha = (alpha / alpha.sum() * NUM_CLASSES).to(device)
    print(f"alpha ({args.alpha_mode}):", [round(a, 3) for a in alpha.tolist()])

    maj = int(freq.argmax())
    maj_acc = (torch.cat([d.joint_type for d in test_ds]) == maj).float().mean().item()
    print(f"majority ({JOINT_TYPES[maj]}) test acc: {maj_acc:.4f}  <- beat this")

    model = TypeClassifier(geo_dim=args.geo_dim, dropout=args.dropout).to(device)
    print("params:", sum(p.numel() for p in model.parameters()))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best, best_state, best_ep = -1.0, None, -1
    trace = []
    for ep in range(1, args.epochs + 1):
        model.train(); random.shuffle(train_ds); tot = n = 0
        for data in train_ds:
            data = data.to(device)
            logits = model(data)
            if logits.size(0) == 0:
                continue
            loss = focal_loss(logits, data.joint_type, alpha, args.gamma)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach()) * data.joint_type.size(0)
            n += data.joint_type.size(0)
        train_loss = tot / max(n, 1)
        row = {"epoch": ep, "train_loss": round(train_loss, 5)}
        if ep % args.eval_every == 0 or ep == 1:
            vm = evaluate(model, val_ds, device)
            flag = ""
            if vm["macro_f1"] > best:
                best, best_ep = vm["macro_f1"], ep
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
                flag = "  *best*"
            row.update(val_acc=round(vm["acc"], 4),
                       val_macro_f1=round(vm["macro_f1"], 4), is_best=bool(flag))
            print(f"ep {ep:>3}  loss {train_loss:.4f}  "
                  f"val acc {vm['acc']:.3f}  val mF1 {vm['macro_f1']:.3f}{flag}")
            if ep - best_ep > args.patience:
                trace.append(row); print("early stop"); break
        trace.append(row)

    model.load_state_dict(best_state); model.to(device)
    tm = evaluate(model, test_ds, device)
    print_eval(f"V3-CLASSIFY OFFICIAL TEST (best epoch {best_ep})", tm)
    print(f"\nmajority {maj_acc:.4f} | V3-classify acc {tm['acc']:.4f} "
          f"macro-F1 {tm['macro_f1']:.4f}   [V2 baseline macro-F1 = 0.253]")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "seed": args.seed, "best_epoch": best_ep,
        "test_acc": tm["acc"], "test_macro_f1": tm["macro_f1"],
        "majority_acc": maj_acc, "v2_baseline_macro_f1": 0.253,
        "per_class_f1": {t: round(float(tm["f1"][i]), 4)
                         for i, t in enumerate(JOINT_TYPES)},
        "support": {t: int(tm["support"][i]) for i, t in enumerate(JOINT_TYPES)},
        "confusion_matrix": tm["cm"].tolist(),
        "config": vars(args), "trace": trace,
    }, indent=1))
    torch.save({"model": best_state, "config": vars(args)}, out.with_suffix(".ckpt"))
    # trace CSV
    cols = ["epoch", "train_loss", "val_acc", "val_macro_f1", "is_best"]
    lines = [",".join(cols)]
    for r in trace:
        lines.append(",".join("" if r.get(c) is None else str(r.get(c, ""))
                              for c in cols))
    out.with_name(out.stem + f"_trace_seed{args.seed}.csv").write_text("\n".join(lines))
    print("wrote", out)


if __name__ == "__main__":
    main()
