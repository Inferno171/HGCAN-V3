"""
data/step_graph_v3.py
STEP -> heterogeneous (face + edge) B-rep graph -> PyG Data.   [HGCAN_V3]

WHY V3 EXISTS
-------------
V1/V2 used a FACE-ONLY face-adjacency graph (data/step_graph.py). For the
joint-LOCALISATION task (JoinABLe-style) that is fatal: a large share of
revolute/cylindrical joints are authored on a *circular edge* (a hole rim),
which is not a node in a face-only graph -> the correct answer is not in the
candidate set. V3 promotes EDGES to first-class nodes alongside faces, exactly
as JoinABLe does, and additionally extracts each entity's *analytic axis* from
its own surface/curve geometry so the joint axis can be READ OFF the matched
entity rather than regressed.

WHAT THIS FILE PRODUCES  (PyG Data)
-----------------------------------
  x              (N, NODE_FEAT_DIM)  unified node features (faces + edges)
  node_type      (N,)                0 = face, 1 = edge
  edge_index     (2, E)              relational graph (both directions)
  edge_type      (E,)               0 cvx / 1 ccv / 2 smooth (face-face)
                                     3 incidence (face-edge)
  entity_axis    (N, 6)             [loc_xyz | dir_xyz] from the entity's OWN
                                     geometry.  NOT a model input by default
                                     (kept for axis derivation at inference).
  entity_axis_valid (N,)            bool; True where an analytic axis exists
                                     (plane normal, cyl/cone axis, line/circle).
  face_uvgrid    (F, 7, 10, 10)     OPTIONAL (use_uvgrid=True) UV-Net face grid
  edge_ugrid     (E, 6, 10)         OPTIONAL (use_uvgrid=True) edge 1D grid

DESIGN NOTES
------------
* Face features reuse the validated V2 19-dim vector verbatim, so V3 with
  use_uvgrid=False is a clean superset of V2's face representation -> any
  localisation result is attributable to the *graph + head*, not the features.
* entity_axis is the leakage-safe primitive: it is computed ONLY from the
  face's/edge's own surface (BRepAdaptor), NEVER from joint_motion /
  joint_axis / geometry_or_origin in assembly.json. Those remain inputs-
  forbidden; here we read geometry the part itself defines.
* UV-grids are OFF by default. They help matching marginally and axis accuracy
  not at all (axis comes from entity_axis). Treat them as an ablation row, not
  the V3 baseline. Consuming face_uvgrid/edge_ugrid needs a small CNN branch
  (separate module) -- this file only extracts them.

API VERIFICATION
----------------
occwl calls mirror data/step_graph.py (already verified on this machine).
The OCC.Core adaptor calls (axis extraction, edge->face ancestry, UV sampling
via occwl.uvgrid) are STANDARD pythonocc-core, but the exact attribute names
vary slightly across pythonocc versions. Every OCC block is wrapped so a miss
degrades to "no axis / no grid" rather than crashing the cache build. Run the
__main__ probe on one STEP file and check the printed counts before trusting
a full rebuild.

Run locally (Windows, hgcan conda env):
  python -m data.step_graph_v3 C:\\path\\to\\<guid>.step
  python -m data.step_graph_v3 C:\\path\\to\\<guid>.step --uvgrid
"""

import math
import sys

import numpy as np
import torch
from torch_geometric.data import Data

try:
    from occwl.compound import Compound
    from occwl.graph import face_adjacency
    from occwl.edge_data_extractor import EdgeDataExtractor, EdgeConvexity
    _HAS_OCCWL = True
except ImportError:
    # occwl/pythonocc is only needed for PREPROCESSING (building the cache).
    # Training/inference load a prebuilt cache and never touch CAD, so on
    # environments without the CAD stack (e.g. Kaggle) we still want this module
    # to import for its constants (NODE_FEAT_DIM, NUM_RELATIONS_V3). The CAD
    # functions below will raise clearly only if actually called.
    _HAS_OCCWL = False
    Compound = face_adjacency = EdgeDataExtractor = EdgeConvexity = None

# ---------------------------------------------------------------------------
# Self-contained face vocabulary + feature extractor (no V2 dependency).
# These are the verified 19-dim face features: 12 surface one-hot + 7 scalars.
# ---------------------------------------------------------------------------
SURF_TYPES = [
    "plane", "cylinder", "cone", "sphere", "torus", "bezier",
    "bspline", "revolution", "extrusion", "offset", "other", "unknown",
]
SURF_TO_IDX = {s: i for i, s in enumerate(SURF_TYPES)}

EDGE_CONVEX, EDGE_CONCAVE, EDGE_SMOOTH = 0, 1, 2
# This mapping dereferences occwl's EdgeConvexity enum, so only build it when
# the CAD stack is present. On cache-only environments (Kaggle) it stays empty;
# it is used exclusively inside preprocessing functions, never during training.
if _HAS_OCCWL:
    CONVEXITY_TO_REL = {
        EdgeConvexity.CONVEX: EDGE_CONVEX,
        EdgeConvexity.CONCAVE: EDGE_CONCAVE,
        EdgeConvexity.SMOOTH: EDGE_SMOOTH,
    }
else:
    CONVEXITY_TO_REL = {}

SMOOTH_TOL_RADS = 0.0872   # ~5 deg: dihedral below this counts as smooth/tangent
CURV_SAMPLES = 5           # 5x5 interior UV grid for curvature statistics


class StepGraphError(Exception):
    """Raised when a body cannot be converted. Message = rejection reason."""


def _face_features(face, total_area):
    """One face -> 19-dim vector: 12 surface one-hot + 7 scalars.

    [0:12]  surface-type one-hot
    [12]    log(area + eps)              (absolute size cue)
    [13]    area / total_area            (scale-invariant share)
    [14]    num_wires - 1                (inner loops = through-holes/bores)
    [15:19] mean/std of gaussian + mean curvature over a UV grid (NaN-safe)
    """
    onehot = np.zeros(len(SURF_TYPES), dtype=np.float32)
    onehot[SURF_TO_IDX.get(face.surface_type(), SURF_TO_IDX["unknown"])] = 1.0

    area = max(face.area(), 0.0)
    log_area = math.log(area + 1e-9)
    rel_area = area / (total_area + 1e-9)
    holes = float(face.num_wires() - 1)

    bounds = face.uv_bounds()
    (umin, vmin), (umax, vmax) = bounds.min_point(), bounds.max_point()
    gauss, mean = [], []
    for u in np.linspace(umin, umax, CURV_SAMPLES + 2)[1:-1]:
        for v in np.linspace(vmin, vmax, CURV_SAMPLES + 2)[1:-1]:
            try:
                gauss.append(face.gaussian_curvature((u, v)))
                mean.append(face.mean_curvature((u, v)))
            except Exception:
                pass
    gauss = np.nan_to_num(np.asarray(gauss, dtype=np.float32))
    mean = np.nan_to_num(np.asarray(mean, dtype=np.float32))
    curv = [
        float(gauss.mean()) if gauss.size else 0.0,
        float(gauss.std()) if gauss.size else 0.0,
        float(mean.mean()) if mean.size else 0.0,
        float(mean.std()) if mean.size else 0.0,
    ]
    curv = [max(-1e3, min(1e3, c)) for c in curv]   # clamp knife-edge b-splines
    return np.concatenate([onehot, [log_area, rel_area, holes], curv]).astype(np.float32)

# --- edge (curve) vocabulary, occwl Edge.curve_type() strings -----------------
CURVE_TYPES = [
    "line", "circle", "ellipse", "hyperbola", "parabola",
    "bezier", "bspline", "offset", "other",
]
CURVE_TO_IDX = {c: i for i, c in enumerate(CURVE_TYPES)}

# --- relations -----------------------------------------------------------------
# 0,1,2 inherited (face-face convexity). 3 is the new face<->edge incidence.
REL_INCIDENCE = 3
NUM_RELATIONS_V3 = 4

# --- node feature layout -------------------------------------------------------
#   [0:2]                node-type one-hot   [is_face, is_edge]
#   [2:2+19]             face block (V2 19-dim) ; zeros on edge nodes
#   [2+19 : ...]         edge block            ; zeros on face nodes
FACE_BLOCK = len(SURF_TYPES) + 7                 # 19
EDGE_BLOCK = len(CURVE_TYPES) + 3                # curve one-hot + 3 scalars
NODE_FEAT_DIM = 2 + FACE_BLOCK + EDGE_BLOCK

UV_NUM = 10   # 10x10 face grid / 10-pt edge grid (UV-Net default)


# ============================================================ axis extraction
def _face_axis(face):
    """Analytic axis of a face from its OWN surface. Returns (valid, loc, dir).

    plane    -> origin on the plane + its normal
    cylinder -> axis location + axis direction
    cone     -> apex-ward axis location + axis direction
    else     -> (False, zeros, zeros)   (sphere/torus/bspline have no single axis)
    """
    try:
        from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
        from OCC.Core.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone
        ad = BRepAdaptor_Surface(face.topods_shape())
        t = ad.GetType()
        if t == GeomAbs_Plane:
            pl = ad.Plane(); ax = pl.Axis()
        elif t == GeomAbs_Cylinder:
            ax = ad.Cylinder().Axis()
        elif t == GeomAbs_Cone:
            ax = ad.Cone().Axis()
        else:
            return False, np.zeros(3, np.float32), np.zeros(3, np.float32)
        loc, d = ax.Location(), ax.Direction()
        return (True,
                np.array([loc.X(), loc.Y(), loc.Z()], np.float32),
                np.array([d.X(), d.Y(), d.Z()], np.float32))
    except Exception:
        return False, np.zeros(3, np.float32), np.zeros(3, np.float32)


def _edge_axis(edge):
    """Analytic axis of an edge. circle -> (centre, normal); line -> (point, dir)."""
    try:
        from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
        from OCC.Core.GeomAbs import GeomAbs_Line, GeomAbs_Circle, GeomAbs_Ellipse
        ad = BRepAdaptor_Curve(edge.topods_shape())
        t = ad.GetType()
        if t == GeomAbs_Circle:
            c = ad.Circle(); ax = c.Axis()
            loc, d = ax.Location(), ax.Direction()
        elif t == GeomAbs_Ellipse:
            c = ad.Ellipse(); ax = c.Axis()
            loc, d = ax.Location(), ax.Direction()
        elif t == GeomAbs_Line:
            ln = ad.Line()
            loc, d = ln.Location(), ln.Direction()
        else:
            return False, np.zeros(3, np.float32), np.zeros(3, np.float32)
        return (True,
                np.array([loc.X(), loc.Y(), loc.Z()], np.float32),
                np.array([d.X(), d.Y(), d.Z()], np.float32))
    except Exception:
        return False, np.zeros(3, np.float32), np.zeros(3, np.float32)


# ============================================================ edge features
def _edge_features(edge, total_len):
    """One edge -> EDGE_BLOCK-dim vector.

    [0:C]   curve-type one-hot
    [C]     log(length + eps)
    [C+1]   length / total_len      (scale-invariant share)
    [C+2]   closed flag             (full circles / closed b-splines = bores)
    """
    onehot = np.zeros(len(CURVE_TYPES), np.float32)
    try:
        ct = edge.curve_type()
    except Exception:
        ct = "other"
    onehot[CURVE_TO_IDX.get(ct, CURVE_TO_IDX["other"])] = 1.0

    try:
        length = max(float(edge.length()), 0.0)
    except Exception:
        length = 0.0
    log_len = math.log(length + 1e-9)
    rel_len = length / (total_len + 1e-9)

    closed = 0.0
    for meth in ("closed_edge", "closed_curve", "closed"):
        if hasattr(edge, meth):
            try:
                closed = float(bool(getattr(edge, meth)()))
                break
            except Exception:
                pass
    return np.concatenate([onehot, [log_len, rel_len, closed]]).astype(np.float32)


# ============================================================ UV-grids (optional)
def _uvgrid_face(face):
    """(7, 10, 10) face grid: 3 point + 3 normal + 1 visibility mask. occwl.uvgrid."""
    from occwl.uvgrid import uvgrid
    pts = np.asarray(uvgrid(face, UV_NUM, UV_NUM, method="point"), np.float32)
    nrm = np.asarray(uvgrid(face, UV_NUM, UV_NUM, method="normal"), np.float32)
    msk = np.asarray(uvgrid(face, UV_NUM, UV_NUM, method="visibility_status"), np.float32)
    msk = (msk > 0).astype(np.float32).reshape(UV_NUM, UV_NUM, 1)
    grid = np.concatenate([pts, nrm, msk], axis=-1)        # (10,10,7)
    return np.transpose(grid, (2, 0, 1))                   # (7,10,10)


def _ugrid_edge(edge):
    """(6, 10) edge grid: 3 point + 3 tangent. occwl.uvgrid.ugrid."""
    from occwl.uvgrid import ugrid
    pts = np.asarray(ugrid(edge, UV_NUM, method="point"), np.float32)
    tan = np.asarray(ugrid(edge, UV_NUM, method="tangent"), np.float32)
    grid = np.concatenate([pts, tan], axis=-1)             # (10,6)
    return np.transpose(grid, (1, 0))                      # (6,10)


# ============================================================ surface samples
SAMPLE_FACE_N = 3       # 3x3 interior UV grid -> 9 points per face
SAMPLE_EDGE_N = 5       # 5 points along each edge


def _face_samples(face):
    """(<=9, 3) surface points in body-local mm, from interior UV grid."""
    try:
        b = face.uv_bounds()
        (umin, vmin), (umax, vmax) = b.min_point(), b.max_point()
        pts = []
        for u in np.linspace(umin, umax, SAMPLE_FACE_N + 2)[1:-1]:
            for v in np.linspace(vmin, vmax, SAMPLE_FACE_N + 2)[1:-1]:
                try:
                    pts.append(np.asarray(face.point((float(u), float(v))), np.float32))
                except Exception:
                    pass
        return np.stack(pts) if pts else np.zeros((0, 3), np.float32)
    except Exception:
        return np.zeros((0, 3), np.float32)


def _edge_samples(edge):
    """(<=5, 3) points along the curve in body-local mm (BRepAdaptor_Curve)."""
    try:
        from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
        ad = BRepAdaptor_Curve(edge.topods_shape())
        t0, t1 = ad.FirstParameter(), ad.LastParameter()
        pts = []
        for s in np.linspace(t0, t1, SAMPLE_EDGE_N):
            p = ad.Value(float(s))
            pts.append(np.array([p.X(), p.Y(), p.Z()], np.float32))
        return np.stack(pts) if pts else np.zeros((0, 3), np.float32)
    except Exception:
        return np.zeros((0, 3), np.float32)


# ============================================================ edge<->face ancestry
def _edge_face_map(shape, faces):
    """For each TopoDS edge, the indices (into `faces`) of the faces it bounds.
    LINEAR (O(E+F)): walk each face's edges once and accumulate parents in a
    shape-keyed dict. Replaces the O(E*F) nested-IsSame scan, which grinds for
    minutes on high-edge-count parts (e.g. breadboards with hundreds of holes).
    Includes boundary edges of open shells (hole-rim circles that matter).
    """
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_EDGE
    from OCC.Core.TopoDS import topods
    from occwl.edge import Edge

    # TopoDS_Shape is hashable in pythonocc (by TShape+location); use it as key.
    edge_rep = {}      # key -> representative TopoDS_Edge (first seen)
    edge_faces = {}    # key -> set of face indices
    for k, f in enumerate(faces):
        fexp = TopExp_Explorer(f.topods_shape(), TopAbs_EDGE)
        while fexp.More():
            e = topods.Edge(fexp.Current())
            key = e.__hash__()                 # cheap, version-stable shape hash
            if key not in edge_rep:
                edge_rep[key] = e
                edge_faces[key] = set()
            edge_faces[key].add(k)
            fexp.Next()

    out = {}
    for ei, (key, e) in enumerate(edge_rep.items()):
        out[ei] = (Edge(e), sorted(edge_faces[key]))
    return out


# ============================================================ main builder
def solid_to_graph_v3(shape, use_uvgrid=False) -> Data:
    """occwl Solid/Shell/Compound -> heterogeneous PyG Data (faces + edges)."""
    try:
        nxg = face_adjacency(shape, self_loops=False)
    except RuntimeError as e:
        raise StepGraphError(f"non-manifold: {e}")
    if nxg is None:
        raise StepGraphError("open/non-manifold shell (face_adjacency returned None)")
    if nxg.number_of_nodes() == 0:
        raise StepGraphError("zero faces")

    # ---- faces: keep V2 ordering EXACTLY (EntityMapper / sorted nodes) ----
    faces = [nxg.nodes[i]["face"] for i in sorted(nxg.nodes)]
    F = len(faces)
    total_area = sum(max(f.area(), 0.0) for f in faces)

    feats = np.zeros((F, NODE_FEAT_DIM), np.float32)
    node_type = np.zeros(F, np.int64)
    axis = np.zeros((F, 6), np.float32)
    axis_ok = np.zeros(F, bool)
    face_samp_pts, face_samp_eid = [], []
    for k, f in enumerate(faces):
        feats[k, 0] = 1.0                                  # is_face
        feats[k, 2:2 + FACE_BLOCK] = _face_features(f, total_area)
        ok, loc, d = _face_axis(f)
        axis_ok[k] = ok
        if ok:
            axis[k, :3], axis[k, 3:] = loc, d
        sp = _face_samples(f)
        if sp.shape[0]:
            face_samp_pts.append(sp); face_samp_eid.append(np.full(sp.shape[0], k, np.int64))

    face_uv = [_uvgrid_face(f) for f in faces] if use_uvgrid else None

    # ---- face-face convexity edges (same logic as V2) ----
    src, dst, rel = [], [], []
    seen = {}
    for i, j, attrs in nxg.edges(data=True):
        key = (min(i, j), max(i, j))
        if key in seen:
            r = seen[key]
        else:
            ext = EdgeDataExtractor(attrs["edge"], [faces[i], faces[j]], num_samples=10)
            r = (CONVEXITY_TO_REL[ext.edge_convexity(SMOOTH_TOL_RADS)]
                 if ext.good else EDGE_SMOOTH)
            seen[key] = r
        # sorted(nxg.nodes) may not be 0..F-1; remap to compact face indices
        src.append(i); dst.append(j); rel.append(r)
    # remap original nx node ids -> compact 0..F-1
    id2idx = {nid: k for k, nid in enumerate(sorted(nxg.nodes))}
    src = [id2idx[s] for s in src]; dst = [id2idx[d] for d in dst]

    # ---- edges as nodes + face-edge incidence ----
    efmap = _edge_face_map(shape, faces)
    edge_objs, edge_fidx = [], []
    for _, (eobj, fidx) in efmap.items():
        edge_objs.append(eobj); edge_fidx.append(fidx)
    E = len(edge_objs)
    total_len = sum(max(getattr(e, "length", lambda: 0.0)(), 0.0)
                    for e in edge_objs) if E else 0.0

    edge_feats = np.zeros((E, NODE_FEAT_DIM), np.float32)
    edge_axis = np.zeros((E, 6), np.float32)
    edge_axis_ok = np.zeros(E, bool)
    edge_samp_pts, edge_samp_eid = [], []
    for k, e in enumerate(edge_objs):
        edge_feats[k, 1] = 1.0                             # is_edge
        edge_feats[k, 2 + FACE_BLOCK:] = _edge_features(e, total_len)
        ok, loc, d = _edge_axis(e)
        edge_axis_ok[k] = ok
        if ok:
            edge_axis[k, :3], edge_axis[k, 3:] = loc, d
        sp = _edge_samples(e)
        if sp.shape[0]:
            edge_samp_pts.append(sp); edge_samp_eid.append(np.full(sp.shape[0], F + k, np.int64))

    edge_uv = [_ugrid_edge(e) for e in edge_objs] if use_uvgrid else None

    # incidence links: face_idx <-> (F + edge_idx), both directions, rel=3
    for ei, fidx in enumerate(edge_fidx):
        enode = F + ei
        for fi in fidx:
            src += [fi, enode]; dst += [enode, fi]
            rel += [REL_INCIDENCE, REL_INCIDENCE]

    # ---- assemble ----
    x = np.concatenate([feats, edge_feats], axis=0)        # (F+E, D)
    nt = np.concatenate([node_type, np.ones(E, np.int64)])
    ent_axis = np.concatenate([axis, edge_axis], axis=0)
    ent_ok = np.concatenate([axis_ok, edge_axis_ok], axis=0)

    if not np.isfinite(x).all():
        raise StepGraphError("non-finite node features")

    if src:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_type = torch.tensor(rel, dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_type = torch.zeros((0,), dtype=torch.long)

    data = Data(
        x=torch.from_numpy(x),
        edge_index=edge_index,
        edge_type=edge_type,
    )
    data.node_type = torch.from_numpy(nt)
    data.entity_axis = torch.from_numpy(ent_axis)
    data.entity_axis_valid = torch.from_numpy(ent_ok)
    data.n_faces = F
    data.n_edges = E
    # per-entity surface sample points (body-local mm) + their entity id (0..F+E-1)
    all_sp = face_samp_pts + edge_samp_pts
    all_eid = face_samp_eid + edge_samp_eid
    if all_sp:
        data.entity_samples = torch.from_numpy(np.concatenate(all_sp, 0))
        data.entity_sample_eid = torch.from_numpy(np.concatenate(all_eid, 0))
    else:
        data.entity_samples = torch.zeros((0, 3), dtype=torch.float32)
        data.entity_sample_eid = torch.zeros((0,), dtype=torch.long)
    if use_uvgrid:
        data.face_uvgrid = torch.from_numpy(np.stack(face_uv)) if F else torch.zeros((0, 7, UV_NUM, UV_NUM))
        data.edge_ugrid = torch.from_numpy(np.stack(edge_uv)) if E else torch.zeros((0, 6, UV_NUM))
    return data


def _merge_graphs(graphs):
    """Concatenate per-solid graphs into ONE disconnected (islands) graph,
    matching V2's full-solid policy. Re-indexes edges, samples, node counts.
    Faces stay ahead of edges WITHIN each solid; offsets accumulate across."""
    if len(graphs) == 1:
        return graphs[0]
    xs, nts, axes, oks = [], [], [], []
    e_src, e_dst, e_rel = [], [], []
    sp_pts, sp_eid = [], []
    n_off = 0                                   # running node offset
    F_tot = E_tot = 0
    for g in graphs:
        n = g.x.size(0)
        xs.append(g.x); nts.append(g.node_type)
        axes.append(g.entity_axis); oks.append(g.entity_axis_valid)
        ei = g.edge_index + n_off
        e_src += ei[0].tolist(); e_dst += ei[1].tolist()
        e_rel += g.edge_type.tolist()
        if g.entity_samples.shape[0]:
            sp_pts.append(g.entity_samples)
            sp_eid.append(g.entity_sample_eid + n_off)
        n_off += n
        F_tot += int(g.n_faces); E_tot += int(g.n_edges)
    data = Data(
        x=torch.cat(xs, 0),
        edge_index=(torch.tensor([e_src, e_dst], dtype=torch.long)
                    if e_src else torch.zeros((2, 0), dtype=torch.long)),
        edge_type=(torch.tensor(e_rel, dtype=torch.long)
                   if e_rel else torch.zeros((0,), dtype=torch.long)),
    )
    data.node_type = torch.cat(nts, 0)
    data.entity_axis = torch.cat(axes, 0)
    data.entity_axis_valid = torch.cat(oks, 0)
    data.n_faces = F_tot
    data.n_edges = E_tot
    data.entity_samples = torch.cat(sp_pts, 0) if sp_pts else torch.zeros((0, 3))
    data.entity_sample_eid = (torch.cat(sp_eid, 0) if sp_eid
                              else torch.zeros((0,), dtype=torch.long))
    return data


def step_to_graph_v3(step_path: str, use_uvgrid=False) -> Data:
    """One a1.0.0 <guid>.step -> heterogeneous PyG Data. ALL solids are kept as
    disconnected islands (V2 full-solid policy), not just the largest."""
    comp = Compound.load_from_step(str(step_path))
    if comp is None:
        raise StepGraphError("STEP read failed")
    solids = list(comp.solids())
    if len(solids) >= 1:
        graphs = []
        for s in solids:
            try:
                graphs.append(solid_to_graph_v3(s, use_uvgrid=use_uvgrid))
            except StepGraphError:
                continue                        # skip a bad island, keep the rest
        if not graphs:
            raise StepGraphError("no usable solids")
        return _merge_graphs(graphs)
    if sum(1 for _ in comp.faces()) == 0:
        raise StepGraphError("no faces in STEP file (empty transfer)")
    return solid_to_graph_v3(comp, use_uvgrid=use_uvgrid)


if __name__ == "__main__":
    use_uv = "--uvgrid" in sys.argv
    path = [a for a in sys.argv[1:] if not a.startswith("--")][0]
    g = step_to_graph_v3(path, use_uvgrid=use_uv)

    nfaces = int((g.node_type == 0).sum())
    nedges = int((g.node_type == 1).sum())
    rels = torch.bincount(g.edge_type, minlength=NUM_RELATIONS_V3).tolist()
    ax_faces = int(g.entity_axis_valid[g.node_type == 0].sum())
    ax_edges = int(g.entity_axis_valid[g.node_type == 1].sum())

    print(f"nodes           : {g.num_nodes}  ({nfaces} faces + {nedges} edges)")
    print(f"x               : {tuple(g.x.shape)}   (NODE_FEAT_DIM={NODE_FEAT_DIM})")
    print(f"edge_index      : {tuple(g.edge_index.shape)}")
    print(f"relations       : cvx={rels[0]} ccv={rels[1]} smooth={rels[2]} "
          f"incidence={rels[3]}")
    print(f"analytic axes   : faces {ax_faces}/{nfaces}   edges {ax_edges}/{nedges}")
    if use_uv:
        print(f"face_uvgrid     : {tuple(g.face_uvgrid.shape)}")
        print(f"edge_ugrid      : {tuple(g.edge_ugrid.shape)}")
    print("\nSANITY: 'analytic axes / edges' should be HIGH (most circles+lines "
          "resolve).\nIf it is ~0, the OCC.Core adaptor import path differs on "
          "your pythonocc\nversion -- fix _edge_axis/_face_axis before rebuilding "
          "the cache.")
