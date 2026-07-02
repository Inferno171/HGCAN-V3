"""
data/assembly_graph_v3.py                                          [HGCAN_V3]
assembly.json (+ per-body V3 graphs) -> AssemblyDataV3 for joint LOCALISATION.

ALIGNED TO THE VERIFIED V2 PIPELINE (data/build_dataset.py):
  * nodes      = geometry-bearing occurrences, sorted-UUID order
  * contacts   = entity_one/two["occurrence"]  (dict OR list)   [V2-verified]
  * body file  = _body_filename(): try keys, fall back <uuid>.step [V2-verified]
  * positions  = physical_properties.center_of_mass, world cm,
                 chain_origin() fallback                          [V2-verified]
  * tree        = tree_depths_and_parents(): parents -> parent/child/sibling
  * joints     = dict, occurrence_one/two, joint_motion.joint_type [V2-verified]

NEW IN V3 (localisation target):
  * entities (faces+edges) are nodes, each with an analytic axis (step_graph_v3)
  * 5 assembly relations: contact / kNN / parent>child / child>parent / sibling
  * per-joint supervision = contributing entity on each side + GT axis

FRAME + UNITS (the crux of geometric labelling):
  entity_axis from step_graph_v3 is in BODY-LOCAL MM.
  joint geometry (geometry_or_origin) is in ASSEMBLY CM, world frame.
  To compare, each occurrence's entities are pushed to WORLD frame with the
  composed occurrence transform (V2's tmat/chain model, extended to carry R),
  and cm translations are bridged to mm (x10). Algebra:
      world_mm = R @ loc_mm + (o_cm * 10)        (loc_mm/10 then *10 cancels)
      dir_world = normalize(R @ dir_local)
  >>> The ONE remaining unknown is whether geometry_or_origin is world-frame
      (assumed) or occurrence-local. label_quality_report() decides: if it is
      occurrence-local, residuals explode and the verdict says so -> then drop
      the world transform on the joint side instead. <<<

SEAMS (marked >>> SEAM n <<<):
  1. geometry_or_origin_{one,two} key names (origin point + axis vector).
  2. geometry_or_origin coordinate frame (world vs occurrence-local).
Both are read out by the self-check; nothing else here is a guess anymore.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

from data.step_graph_v3 import step_to_graph_v3, NODE_FEAT_DIM, NUM_RELATIONS_V3

JOINT_TYPES = ["RigidJointType", "RevoluteJointType", "SliderJointType",
               "CylindricalJointType", "PinSlotJointType", "PlanarJointType",
               "BallJointType"]
JT_TO_IDX = {t: i for i, t in enumerate(JOINT_TYPES)}

REL_CONTACT, REL_KNN, REL_PARENT, REL_CHILD, REL_SIBLING = 0, 1, 2, 3, 4
NUM_ASM_RELATIONS = 5

KNN_K = 8
MM_PER_CM = 10.0
PARALLEL_TOL = 0.985           # |cos| > tol  == axes parallel  (~10 deg)
POS_PARALLEL = 0.996           # tighter, for the multi-positive equivalence set
POS_DIST_MM = 2.0


class AssemblyDataV3(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == "ent_edge_index":
            return self.x_ent.size(0)
        if key in ("ent_to_occ", "asm_edge_index", "joint_occ_pairs"):
            return self.num_occ
        if key in ("joint_pos_i", "joint_pos_j"):
            return self.x_ent.size(0)
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ("ent_edge_index", "asm_edge_index", "joint_occ_pairs"):
            return 1
        return super().__cat_dim__(key, value, *args, **kwargs)


# ============================================================ V2 helpers (reused)
def tree_depths_and_parents(tree):
    depths, parents = {}, {}
    def walk(node, parent, depth):
        for u, ch in (node or {}).items():
            depths[u] = depth; parents[u] = parent
            walk(ch, u, depth + 1)
    walk((tree or {}).get("root", {}), None, 1)
    return depths, parents


def tmat(t):
    o = np.array([t["origin"][k] for k in "xyz"], np.float64)
    R = np.stack([[t[a][k] for k in "xyz"]
                  for a in ("x_axis", "y_axis", "z_axis")], axis=1).astype(np.float64)
    return R, o


def chain_transform(u, parents, occs):
    """Composed WORLD transform (R, o_cm) of an occurrence's local frame."""
    chain = []
    while u is not None:
        chain.append(u); u = parents.get(u)
    R, o = np.eye(3), np.zeros(3)
    for v in reversed(chain):
        t = (occs.get(v) or {}).get("transform")
        if not t:
            continue
        Rv, ov = tmat(t)
        R, o = R @ Rv, R @ ov + o
    return R, o


def chain_origin(u, parents, occs):
    return chain_transform(u, parents, occs)[1]


def _body_filename(body_uuid, body_rec):                  # V2-verified
    for k in ("step", "file_name", "file", "filename"):
        if isinstance(body_rec, dict) and body_rec.get(k):
            return str(body_rec[k])
    return f"{body_uuid}.step"


# ============================================================ joint geometry
def _vec(d, *keys):
    for k in keys:
        v = d.get(k) if isinstance(d, dict) else None
        if isinstance(v, dict) and {"x", "y", "z"} <= set(v):
            return np.array([v["x"], v["y"], v["z"]], np.float64)
    return None


def _entity_anchor(geom):
    """geometry_or_origin -> dict(point, want_edge, axis, origin) | None.
    Label anchor is the nested entity_one.point_on_entity (verified present on
    every joint). axis/origin kept for training-time metrics only."""
    if not isinstance(geom, dict):
        return None
    e = geom.get("entity_one")
    if not isinstance(e, dict):
        return None
    poe = _vec(e, "point_on_entity")
    if poe is None:
        return None
    return {"point": poe,
            "want_edge": e.get("type") == "BRepEdge",
            "axis": _vec(geom, "primary_axis_vector", "axis", "direction"),
            "origin": _vec(geom, "origin", "point")}


# ============================================================ frame discovery
POS_BAND_MM = 1.0          # entities within this of the best sample dist = positives


def _frame_candidates(p_cm, R, o_cm):
    """Body-local mm query points under 4 frame/scale hypotheses. The correct
    convention lands the point ON an entity (distance -> 0); we keep the winner
    and report which won, so the cm/mm and world/local conventions are LEARNED."""
    pl = R.T @ (p_cm - o_cm)               # world(cm) -> occ-local(cm)
    return {
        "raw_x10": p_cm * MM_PER_CM,       # JSON already occ-local, cm->mm
        "inv_x10": pl * MM_PER_CM,         # JSON world -> local, cm->mm
        "raw":     p_cm.astype(np.float64),
        "inv":     pl,
    }


def _label_point(p_cm, R, o_cm, samp_pts, samp_eid, node_type_local, want_edge):
    """Assign the contributing entity by nearest surface sample, auto-selecting
    the frame. samp_eid is OCCURRENCE-LOCAL entity id (0..k-1).
    Returns (best_local_eid, positives, dist_mm, frame) | None."""
    if samp_pts.shape[0] == 0:
        return None
    want_t = 1 if want_edge else 0
    mask = node_type_local[samp_eid] == want_t
    if not mask.any():
        mask = np.ones(samp_eid.shape[0], bool)      # type fallback
    sp, se = samp_pts[mask], samp_eid[mask]

    cands = _frame_candidates(p_cm, R, o_cm)
    best = None                                       # (eid, frame, dist)
    for name, q in cands.items():
        d = np.linalg.norm(sp - q[None, :], axis=1)
        k = int(np.argmin(d))
        if best is None or d[k] < best[2]:
            best = (int(se[k]), name, float(d[k]))
    eid, frame, dist = best

    q = cands[frame]                                  # multi-positive in winner frame
    pos = []
    for u in np.unique(se):
        dd = float(np.linalg.norm(sp[se == u] - q[None, :], axis=1).min())
        if dd < dist + POS_BAND_MM:
            pos.append(int(u))
    if eid not in pos:
        pos.append(eid)
    return eid, pos, dist, frame


# ============================================================ main builder
def build_assembly_v3(json_path, body_graph_loader, knn_k=KNN_K):
    doc = json.loads(Path(json_path).read_text(encoding="utf-8"))
    occs = doc.get("occurrences") or {}
    bodies = doc.get("bodies") or {}
    depths, parents = tree_depths_and_parents(doc.get("tree", {}))

    # ---- nodes: geometry-bearing occurrences, sorted-UUID order (V2) ----
    node_uuids = sorted(u for u, o in occs.items() if (o or {}).get("bodies"))
    if len(node_uuids) < 2:
        raise ValueError("fewer than 2 geometry-bearing occurrences")
    idx = {u: i for i, u in enumerate(node_uuids)}
    N = len(node_uuids)

    # ---- per-node: entities + world transform + surface samples ----
    ent_x, e_src, e_dst, e_rel = [], [], [], []
    ent_to_occ, ent_axis, ent_valid, node_type = [], [], [], []
    slice_of, world_R, world_o = {}, {}, {}
    occ_samp_pts, occ_samp_eid = {}, {}          # per-occ (S,3) mm, (S,) local eid
    m = 0
    occ_has_geom = []
    for u in node_uuids:
        i = idx[u]
        world_R[i], world_o[i] = chain_transform(u, parents, occs)
        start = m
        local_off = 0                            # entity offset within this occ
        sp_list, se_list = [], []
        for buid in ((occs[u] or {}).get("bodies") or {}):
            g = body_graph_loader(buid, _body_filename(buid, bodies.get(buid, {})))
            if g is None or g.num_nodes == 0:
                continue
            f = g.x.size(0)
            ent_x.append(g.x.numpy())
            ei = g.edge_index.numpy() + m
            e_src += ei[0].tolist(); e_dst += ei[1].tolist()
            e_rel += g.edge_type.numpy().tolist()
            ent_to_occ += [i] * f
            ent_axis.append(g.entity_axis.numpy())
            ent_valid.append(g.entity_axis_valid.numpy())
            node_type.append(g.node_type.numpy())
            if hasattr(g, "entity_samples") and g.entity_samples.shape[0]:
                sp_list.append(g.entity_samples.numpy())
                se_list.append(g.entity_sample_eid.numpy() + local_off)
            m += f
            local_off += f
        slice_of[i] = (start, m)
        occ_has_geom.append(m > start)
        occ_samp_pts[i] = np.concatenate(sp_list, 0) if sp_list else np.zeros((0, 3), np.float32)
        occ_samp_eid[i] = np.concatenate(se_list, 0) if se_list else np.zeros((0,), np.int64)

    if m == 0:
        raise ValueError("no entities loaded for any occurrence")

    x_ent = torch.from_numpy(np.concatenate(ent_x, 0))
    entity_axis = np.concatenate(ent_axis, 0)
    entity_valid = np.concatenate(ent_valid, 0)
    node_type_np = np.concatenate(node_type, 0)

    # ---- positions: world com (cm), chain-origin fallback (V2) ----
    P = np.zeros((N, 3))
    for u, i in idx.items():
        com = ((occs[u] or {}).get("physical_properties") or {}).get("center_of_mass")
        P[i] = [com["x"], com["y"], com["z"]] if com else chain_origin(u, parents, occs)

    # ---- assembly edges ----
    asrc, adst, arel = [], [], []
    def add(ia, ib, r):
        if ia != ib:
            asrc.extend([ia, ib]); adst.extend([ib, ia]); arel.extend([r, r])

    # contact: entity.occurrence  (dict OR list)  [V2-verified]
    contacts = doc.get("contacts") or []
    for c in (contacts.values() if isinstance(contacts, dict) else contacts):
        try:
            a = c["entity_one"]["occurrence"]; b = c["entity_two"]["occurrence"]
        except (KeyError, TypeError):
            continue
        if a in idx and b in idx and a != b:
            add(idx[a], idx[b], REL_CONTACT)

    # kNN(8) over world com
    geo = [i for i in range(N) if occ_has_geom[i]]
    if len(geo) > 1:
        Pg = np.stack([P[i] for i in geo])
        for a_, ia in enumerate(geo):
            order = np.argsort(np.linalg.norm(Pg - Pg[a_], axis=1))[1:knn_k + 1]
            for b_ in order:
                add(ia, geo[b_], REL_KNN)

    # tree: parents -> directional parent/child + sibling
    children_of = {}
    for child, par in parents.items():
        if par in idx and child in idx:
            asrc.extend([idx[par], idx[child]])
            adst.extend([idx[child], idx[par]])
            arel.extend([REL_PARENT, REL_CHILD])
            children_of.setdefault(par, []).append(child)
    for par, kids in children_of.items():
        for a_ in range(len(kids)):
            for b_ in range(a_ + 1, len(kids)):
                add(idx[kids[a_]], idx[kids[b_]], REL_SIBLING)

    # ---- supervision: joints -> contributing entities (point containment) ----
    jp_i, jp_j, jt = [], [], []
    jpos_i, jpos_j, jaxis, residuals = [], [], [], []
    for j in (doc.get("joints") or {}).values():
        j = j or {}
        o1, o2 = j.get("occurrence_one"), j.get("occurrence_two")
        if o1 not in idx or o2 not in idx or o1 == o2:
            continue
        a1 = _entity_anchor(j.get("geometry_or_origin_one"))
        a2 = _entity_anchor(j.get("geometry_or_origin_two"))
        if a1 is None or a2 is None:
            continue
        i1, i2 = idx[o1], idx[o2]
        s1, e1 = slice_of[i1]; s2, e2 = slice_of[i2]
        if e1 <= s1 or e2 <= s2:
            continue
        jtype = (j.get("joint_motion") or {}).get("joint_type")

        def side(occ_i, anchor):
            s, e = slice_of[occ_i]
            return _label_point(anchor["point"], world_R[occ_i], world_o[occ_i],
                                occ_samp_pts[occ_i], occ_samp_eid[occ_i],
                                node_type_np[s:e], anchor["want_edge"]), s

        r1, s1 = side(i1, a1)
        r2, s2 = side(i2, a2)
        if r1 is None or r2 is None:
            continue
        eid1, pos1, d1, f1 = r1
        eid2, pos2, d2, f2 = r2
        jp_i.append(i1); jp_j.append(i2); jt.append(JT_TO_IDX.get(jtype, 0))
        jpos_i.append(torch.tensor([s1 + p for p in pos1], dtype=torch.long))
        jpos_j.append(torch.tensor([s2 + p for p in pos2], dtype=torch.long))
        # GT axis for training metrics (JSON frame; origin cm->mm)
        ax = a1["axis"] if a1["axis"] is not None else np.array([0., 0., 1.])
        org = (a1["origin"] if a1["origin"] is not None else a1["point"]) * MM_PER_CM
        jaxis.append(np.concatenate([org, ax]).astype(np.float32))
        residuals.append((d1, f1)); residuals.append((d2, f2))

    if not jp_i:
        raise ValueError("no jointed pair survived entity labelling")

    edge_index = torch.tensor([e_src, e_dst], dtype=torch.long) if e_src \
        else torch.zeros((2, 0), dtype=torch.long)
    edge_type = torch.tensor(e_rel, dtype=torch.long) if e_rel \
        else torch.zeros((0,), dtype=torch.long)
    a_idx = torch.tensor([asrc, adst], dtype=torch.long) if asrc \
        else torch.zeros((2, 0), dtype=torch.long)
    a_typ = torch.tensor(arel, dtype=torch.long) if arel \
        else torch.zeros((0,), dtype=torch.long)

    d = AssemblyDataV3(x_ent=x_ent)
    d.ent_edge_index = edge_index
    d.ent_edge_type = edge_type
    d.ent_to_occ = torch.tensor(ent_to_occ, dtype=torch.long)
    d.node_type = torch.from_numpy(node_type_np)
    d.entity_axis = torch.from_numpy(entity_axis.astype(np.float32))
    d.entity_axis_valid = torch.from_numpy(entity_valid)
    d.num_occ = N
    d.asm_edge_index = a_idx
    d.asm_edge_type = a_typ
    d.occ_has_geom = torch.tensor(occ_has_geom, dtype=torch.bool)
    d.joint_occ_pairs = torch.tensor([jp_i, jp_j], dtype=torch.long)
    d.joint_type = torch.tensor(jt, dtype=torch.long)
    d.joint_axis_gt = torch.from_numpy(np.stack(jaxis))
    d.joint_pos_i = jpos_i
    d.joint_pos_j = jpos_j
    d.assembly_id = Path(json_path).parent.name
    d.occ_uuids = node_uuids
    d._label_residuals = residuals
    return d


# ============================================================ the gate
def label_quality_report(residuals):
    """residuals: list of (point_dist_mm, frame_name). Decides go/no-go AND
    reports which frame/scale convention won, so the cm/mm + world/local
    question is answered by the data instead of asserted."""
    if not residuals:
        print("no residuals — 0 joints labelled. Check entity_one.point_on_entity "
              "is present and occurrences carry geometry.")
        return
    from collections import Counter
    dist = np.array([r[0] for r in residuals], float)
    fc = Counter(r[1] for r in residuals)
    def pct(q): return float(np.percentile(dist, q))
    print(f"labelled sides     : {dist.size}")
    print(f"point residual (mm): p50 {pct(50):.3f}  p90 {pct(90):.3f}  "
          f"max {dist.max():.3f}")
    print(f"frame won (auto)   : " +
          ", ".join(f"{k}={v}" for k, v in fc.most_common()))
    c1 = float((dist < 1.0).mean()); c5 = float((dist < 5.0).mean())
    print(f"within 1mm: {c1*100:.1f}%    within 5mm: {c5*100:.1f}%")
    if pct(50) < 2.0 and c5 > 0.7:
        dom = fc.most_common(1)[0][0]
        print(f"VERDICT: labels trustworthy (dominant frame '{dom}') -> train.")
    else:
        print("VERDICT: residuals large -> point_on_entity not landing on your "
              "geometry. Either the right frame isn't in _frame_candidates, or "
              "sample density is too low. Paste this report and we adjust.")
