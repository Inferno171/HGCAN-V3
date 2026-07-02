"""
build_cache_v3.py                                                  [HGCAN_V3]
Preprocessing driver across all 11 shards. Body-file resolution and JSON
parsing now live in build_assembly_v3 (aligned to V2); this script only
indexes the shards, prefilters to jointed assemblies, runs the gate, and
writes the cache + splits.

ORDER (do not skip the gate):
  1) gate dry run (writes nothing):
       python -m build_cache_v3 --raw-dir D:\\...\\Fusion360 ^
           --split D:\\...\\train_test.json --out cache_v3 --sample 200
  2) full build:
       python -m build_cache_v3 --raw-dir D:\\...\\Fusion360 ^
           --split D:\\...\\train_test.json --out cache_v3

--raw-dir is the PARENT holding a1.0.0_00 .. a1.0.0_10. One glob recurses all
shards, so the partitioning is invisible. Body step files are resolved by
build_assembly_v3 as <asm_dir>/<uuid>.step (V2 convention); if the diagnostic
showed them nested, fix _body_filename usage there, not here.
"""

import argparse
import json
from pathlib import Path

import torch

from data.step_graph_v3 import step_to_graph_v3
from data.assembly_graph_v3 import build_assembly_v3, label_quality_report
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

def build_index(raw_dir):
    """assembly-folder-name -> assembly.json path, across all shards."""
    idx = {}
    for jp in Path(raw_dir).glob("**/assembly.json"):
        idx[jp.parent.name] = jp
    return idx


def is_jointed(json_path):
    """Cheap prefilter: non-empty joints dict, no geometry touched."""
    try:
        doc = json.loads(Path(json_path).read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(doc.get("joints"))


def make_loader(asm_dir, body_cache_dir, use_uvgrid=False):
    body_cache_dir.mkdir(parents=True, exist_ok=True)
    def load(body_uuid, step_filename):
        cpath = body_cache_dir / f"{body_uuid}.pt"
        if cpath.exists():
            try:
                return torch.load(cpath, weights_only=False)
            except Exception:
                pass
        spath = Path(asm_dir) / step_filename
        if not spath.exists():
            return None
        print(f"      body {body_uuid[:8]}", flush=True)
        try:
            g = step_to_graph_v3(str(spath), use_uvgrid=use_uvgrid)
        except Exception:
            return None
        torch.save(g, cpath)
        return g
    return load

def load_all_splits(raw_dir):
    """Merge every shard's train_test.json into one {train:[...], test:[...]}."""
    train, test = [], []
    for sp in Path(raw_dir).glob("**/train_test.json"):
        d = json.loads(Path(sp).read_text(encoding="utf-8"))
        train += [str(n) for n in d.get("train", [])]
        test  += [str(n) for n in d.get("test", [])]
    return {"train": sorted(set(train)), "test": sorted(set(test))}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True, help="parent of a1.0.0_00..10")
    ap.add_argument("--split", default=None)
    ap.add_argument("--out", default="cache_v3")
    ap.add_argument("--sample", type=int, default=0,
                    help=">0 : gate-only dry run over first N jointed train asms")
    ap.add_argument("--uvgrid", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    body_cache = out / "bodies"
    index = build_index(args.raw_dir)
    print(f"indexed {len(index)} assemblies across shards")

    split = load_all_splits(args.raw_dir)
    train = [str(n) for n in split.get("train", [])]
    test = [str(n) for n in split.get("test", [])]
    in_train, in_test = set(train), set(test)
    names = (train[:args.sample] if args.sample else train + test)

    if not args.sample:
        (out / "assemblies").mkdir(parents=True, exist_ok=True)
    else:
        print(f"GATE DRY RUN over up to {len(names)} train assemblies "
              f"(nothing written)\n")

    residuals, built, failed, skipped_nojoint = [], [], [], 0
    for k, name in enumerate(names, 1):
        jp = index.get(name)
        if jp is None:
            failed.append((name, "not in shards")); continue
        if not is_jointed(jp):
            skipped_nojoint += 1; continue
        loader = make_loader(jp.parent, body_cache, use_uvgrid=args.uvgrid)
        try:
            data = build_assembly_v3(str(jp), loader)
        except Exception as e:
            failed.append((name, str(e)[:70])); continue
        residuals.extend(getattr(data, "_label_residuals", []))
        if not args.sample:
            #data._label_residuals = None
            torch.save(data, out / "assemblies" / f"{name}.pt")
        built.append(name)
        if k % 50 == 0:
            print(f"  {k}/{len(names)}  built={len(built)} "
                  f"failed={len(failed)} nojoint={skipped_nojoint}")

    print(f"\nbuilt {len(built)}  failed {len(failed)}  "
          f"non-jointed skipped {skipped_nojoint}")
    if failed[:5]:
        print("first failures:", failed[:5])

    print("\n=== LABEL QUALITY GATE ===")
    label_quality_report(residuals)

    if args.sample:
        print("\n(dry run — drop --sample once the verdict is clean.)")
        return

    bs = set(built)
    splits = {"train": [n for n in train if n in bs],
              "test": [n for n in test if n in bs]}
    (out / "splits.json").write_text(json.dumps(splits, indent=1), encoding="utf-8")
    print(f"\nwrote {out/'splits.json'}  "
          f"({len(splits['train'])} train / {len(splits['test'])} test)")


if __name__ == "__main__":
    main()
