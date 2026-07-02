"""
analyze_assembly_features.py                                       [HGCAN_V3]
Survey assembly.json across all shards and report WHAT IS AVAILABLE and WHETHER
IT IS USABLE as a feature -- so feature decisions are made from coverage numbers,
not assumptions. Pure JSON (no torch/occwl); fast.

Run:
  python analyze_assembly_features.py --raw-dir "D:\\...\\Fusion360"
  python analyze_assembly_features.py --raw-dir "D:\\...\\Fusion360" --sample 400
  python analyze_assembly_features.py --raw-dir "D:\\...\\Fusion360" --all   # incl. non-jointed

For each field group it prints:
  - coverage  (% of records that carry the key, non-null)
  - the set of keys actually present (schema discovery)
  - basic stats for numeric candidates (min / p50 / max)
  - a USABILITY tag for the V3 localisation task:
      USABLE   - legitimate input feature (geometry/structure the part defines)
      LEAKAGE  - encodes the answer; never an input
      TARGET   - the supervision signal itself
      META     - identifiers/paths/cosmetic; not a feature
"""

import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------- helpers
def keyset_coverage(records):
    """records: iterable of dicts -> (n, Counter(key->count), Counter(keyset))."""
    n = 0
    kc = Counter()
    ksets = Counter()
    for r in records:
        if not isinstance(r, dict):
            continue
        n += 1
        present = tuple(sorted(k for k, v in r.items() if v not in (None, "", [], {})))
        ksets[present] += 1
        for k in present:
            kc[k] += 1
    return n, kc, ksets


def pct(c, n):
    return f"{100.0 * c / n:5.1f}%" if n else "  n/a"


def stat_line(name, vals):
    a = np.asarray([v for v in vals if v is not None], float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return f"    {name:24} (no values)"
    return (f"    {name:24} min {a.min():10.3f}  p50 {np.median(a):10.3f}  "
            f"max {a.max():10.3f}  n={a.size}")


def header(t):
    print("\n" + "=" * 74)
    print(t)
    print("=" * 74)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--sample", type=int, default=0, help="cap assemblies scanned")
    ap.add_argument("--all", action="store_true", help="include non-jointed assemblies")
    args = ap.parse_args()

    paths = sorted(glob.glob(str(Path(args.raw_dir) / "**" / "assembly.json"),
                             recursive=True))
    print(f"found {len(paths)} assembly.json across shards")

    # accumulators
    n_asm = n_jointed = 0
    toplevel = Counter(); toplevel_null = Counter()
    occ_records = []
    body_records = []
    joint_types = Counter()
    side_entity_type = Counter()
    side_geom_type = Counter()
    entity_keysets = Counter()
    poe_present = axis_present = sec_axis = ter_axis = n_sides = 0
    contact_null = contact_total = 0
    contact_surf_pairs = Counter()
    hole_records = []
    n_holes_total = 0
    props_categories = Counter(); props_industries = Counter()
    props_design = Counter(); props_surftypes = Counter()
    occ_n_bodies = []; occ_grounded = 0; occ_total = 0; occ_with_com = 0
    occ_vol = []; occ_area = []; occ_mass = []; occ_density = []
    tree_depths = []
    contacts_present_count = 0

    for p in paths:
        try:
            d = json.loads(Path(p).read_text(encoding="utf-8"))
        except Exception:
            continue
        joints = d.get("joints") or {}
        if not args.all and not joints:
            continue
        n_asm += 1
        if joints:
            n_jointed += 1

        # top-level
        for k in ("tree", "root", "occurrences", "components", "bodies",
                  "joints", "contacts", "holes", "as_built_joints", "properties"):
            if k in d:
                toplevel[k] += 1
                if d.get(k) in (None, [], {}):
                    toplevel_null[k] += 1

        # occurrences
        occs = d.get("occurrences") or {}
        # tree depth
        par = {}
        def w(node, depth):
            for u, ch in (node or {}).items():
                tree_depths.append(depth); w(ch, depth + 1)
        w((d.get("tree") or {}).get("root", {}), 1)
        for o in occs.values():
            if not isinstance(o, dict):
                continue
            occ_total += 1
            occ_records.append(o)
            occ_n_bodies.append(len(o.get("bodies") or {}))
            if o.get("is_grounded"):
                occ_grounded += 1
            pp = o.get("physical_properties") or {}
            if pp.get("center_of_mass"):
                occ_with_com += 1
            occ_vol.append(pp.get("volume")); occ_area.append(pp.get("area"))
            occ_mass.append(pp.get("mass")); occ_density.append(pp.get("density"))

        # bodies
        for b in (d.get("bodies") or {}).values():
            if isinstance(b, dict):
                body_records.append(b)

        # joints
        for j in joints.values():
            j = j or {}
            joint_types[(j.get("joint_motion") or {}).get("joint_type")] += 1
            for side in ("geometry_or_origin_one", "geometry_or_origin_two"):
                g = j.get(side)
                if not isinstance(g, dict):
                    continue
                n_sides += 1
                side_geom_type[g.get("geometry_type")] += 1
                if "primary_axis_vector" in g: axis_present += 1
                if g.get("secondary_axis_vector"): sec_axis += 1
                if g.get("tertiary_axis_vector"): ter_axis += 1
                e = g.get("entity_one")
                if isinstance(e, dict):
                    side_entity_type[e.get("type")] += 1
                    entity_keysets[tuple(sorted(e.keys()))] += 1
                    if "point_on_entity" in e: poe_present += 1

        # contacts
        c = d.get("contacts")
        contact_total += 1
        if c in (None, [], {}):
            contact_null += 1
        else:
            contacts_present_count += 1
            citer = c.values() if isinstance(c, dict) else c
            for con in citer:
                if not isinstance(con, dict):
                    continue
                s1 = (con.get("entity_one") or {}).get("surface_type")
                s2 = (con.get("entity_two") or {}).get("surface_type")
                if s1 and s2:
                    contact_surf_pairs[tuple(sorted((s1, s2)))] += 1

        # holes
        holes = d.get("holes") or []
        n_holes_total += len(holes)
        for h in holes:
            if isinstance(h, dict):
                hole_records.append(h)

        # assembly properties
        pr = d.get("properties") or {}
        for cat in (pr.get("categories") or []):
            props_categories[cat] += 1
        for ind in (pr.get("industries") or []):
            props_industries[ind] += 1
        if pr.get("design_type"):
            props_design[pr["design_type"]] += 1
        for st in (pr.get("surface_types") or []):
            if isinstance(st, dict):
                props_surftypes[st.get("surface_type")] += st.get("face_count", 0)

        if args.sample and n_asm >= args.sample:
            break

    # ============================================================ REPORT
    print(f"\nscanned {n_asm} assemblies ({n_jointed} jointed)")

    header("TOP-LEVEL KEYS  (presence / null-rate)")
    for k in ("tree", "occurrences", "components", "bodies", "joints",
              "contacts", "holes", "as_built_joints", "properties"):
        present = toplevel.get(k, 0)
        nulls = toplevel_null.get(k, 0)
        print(f"  {k:16} present {pct(present, n_asm)}   "
              f"null-when-present {pct(nulls, max(present,1))}")

    header("OCCURRENCE-LEVEL  (candidate occurrence features -> USABLE)")
    n, kc, _ = keyset_coverage(occ_records)
    for k, c in kc.most_common():
        print(f"  {k:24} {pct(c, n)}")
    print(f"\n  center_of_mass present : {pct(occ_with_com, occ_total)}  (world-frame; kNN + a position feature)")
    print(f"  is_grounded == True    : {pct(occ_grounded, occ_total)}  (boolean structural feature)")
    print(stat_line("n_bodies", occ_n_bodies))
    print(stat_line("volume", occ_vol))
    print(stat_line("area", occ_area))
    print(stat_line("mass", occ_mass))
    print(stat_line("density", occ_density))
    print(stat_line("tree_depth", tree_depths))
    print("  -> these are exactly V2's x_occ inputs; all USABLE (no leakage).")

    header("BODY-LEVEL  (mostly META: file paths; geometry comes from STEP)")
    n, kc, _ = keyset_coverage(body_records)
    for k, c in kc.most_common():
        tag = "META" if k in ("png", "smt", "step", "obj", "name", "type",
                              "appearance", "material") else "USABLE?"
        print(f"  {k:24} {pct(c, n)}   {tag}")

    header("JOINT-LEVEL  (the SUPERVISION; never an input feature)")
    print("  joint_type distribution:")
    for t, c in joint_types.most_common():
        print(f"    {str(t):28} {c:6}  {pct(c, sum(joint_types.values()))}")
    print(f"\n  sides analysed          : {n_sides}")
    print(f"  entity is EDGE          : {pct(side_entity_type.get('BRepEdge',0), n_sides)}")
    print(f"  entity is FACE          : {pct(side_entity_type.get('BRepFace',0), n_sides)}")
    print(f"  point_on_entity present : {pct(poe_present, n_sides)}   (V3 label anchor)")
    print(f"  primary_axis present    : {pct(axis_present, n_sides)}   (axis metric)")
    print(f"  secondary_axis present  : {pct(sec_axis, n_sides)}")
    print(f"  tertiary_axis present   : {pct(ter_axis, n_sides)}")
    print("  geometry_type distribution:")
    for t, c in side_geom_type.most_common(8):
        print(f"    {str(t):40} {c:6}")
    print("  entity key-sets (schema of entity_one):")
    for ks, c in entity_keysets.most_common(4):
        print(f"    {c:6}  {ks}")
    print("\n  -> joint_motion / geometry_or_origin / *_axis_vector = LEAKAGE as input.")
    print("     entity_one + point_on_entity = the V3 TARGET (label only).")

    header("CONTACT-LEVEL  (candidate occurrence-graph relation)")
    print(f"  assemblies with contacts: {pct(contacts_present_count, n_asm)}  "
          f"(null on {pct(contact_null, contact_total)})")
    if contact_surf_pairs:
        print("  top surface-type contact pairs (audit reviewer cyl-cyl claim here):")
        for pair, c in contact_surf_pairs.most_common(8):
            print(f"    {str(pair):48} {c}")

    header("HOLE-LEVEL  (candidate signal; >80% of joints reportedly on holes)")
    print(f"  total holes: {n_holes_total} over {n_asm} assemblies "
          f"({n_holes_total/max(n_asm,1):.1f}/assembly)")
    n, kc, _ = keyset_coverage(hole_records)
    for k, c in kc.most_common():
        print(f"  {k:24} {pct(c, n)}")
    for nk in ("diameter", "length", "depth", "radius"):
        vals = [(h.get(nk)) for h in hole_records if isinstance(h.get(nk), (int, float))]
        if vals:
            print(stat_line(nk, vals))
    print("  -> faces/edges here are .smt indices (NOT your occwl order); use with care.")

    header("ASSEMBLY-LEVEL properties  (global context, optional)")
    print("  design_type:", dict(props_design.most_common(5)))
    print("  top categories:", dict(props_categories.most_common(6)))
    print("  top industries:", dict(props_industries.most_common(6)))
    print("  surface-type face counts (dataset-wide):")
    tot_sf = sum(props_surftypes.values()) or 1
    for st, c in props_surftypes.most_common():
        print(f"    {str(st):28} {c:9}  {pct(c, tot_sf)}")

    header("FEATURE USABILITY SUMMARY  (V3 localisation)")
    rows = [
        ("entity B-rep geometry (STEP)", "USABLE", "faces+edges, surface/curve type, area, curvature -> already in step_graph_v3"),
        ("occurrence physical_properties", "USABLE", "area/vol/mass/density/com -> V2 x_occ; optional occ side-input"),
        ("is_grounded / is_visible / tree_depth / n_bodies", "USABLE", "structural occ features"),
        ("center_of_mass", "USABLE", "world-frame position -> kNN edges + position feature"),
        ("contacts (occurrence pairs)", "USABLE", "graph relation; null on ~19%"),
        ("tree parent/child/sibling", "USABLE*", "structural relation; active on ~11% (mostly flat) -> ablate"),
        ("holes (origin/diameter/depth)", "USABLE*", "strong joint cue but .smt index gap; geometric match only"),
        ("assembly properties (category/bbox)", "USABLE", "coarse global context"),
        ("joint_motion.joint_type", "LEAKAGE/TARGET", "label for typing; never an input"),
        ("geometry_or_origin (origin/axis)", "LEAKAGE", "encodes the joint; input-forbidden"),
        ("entity_one + point_on_entity", "TARGET", "the localisation supervision"),
        ("body png/smt/step/obj/material", "META", "paths/cosmetic; not features"),
    ]
    for name, tag, note in rows:
        print(f"  [{tag:14}] {name}")
        print(f"                   {note}")


if __name__ == "__main__":
    main()
