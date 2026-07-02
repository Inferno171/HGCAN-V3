"""
models/model_v3.py                                                 [HGCAN_V3]
Joint-localisation model. Fresh architecture for the V3 goal (entity match +
axis), NOT the V2 classifier.

Flow:
  x_ent (faces+edges) --EntityEncoder (R-GCN, 4 rel)-->  h_ent  (per ENTITY)
  h_ent --scatter-->  h_occ  (per occurrence, pooled)
  h_occ --AssemblyContext (R-GCN, 5 rel: contact/knn/parent/child/sibling)-->
        h_occ_ctx
  add assembly context back onto each entity:  z_ent = proj(h_ent) + h_occ_ctx[occ]
  for each jointed pair (occ_i, occ_j):
        score E_i x E_j  (bilinear)  ->  match matrix  ->  argmax entity pair
        axis = entity_axis[matched entity]            (read, not regressed)

Heads:
  MatchingHead   : the localisation head (primary).
  TypeHead       : optional 7-class auxiliary on the pooled pair (multi-task).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from torch_geometric.utils import scatter

from data.step_graph_v3 import NODE_FEAT_DIM, NUM_RELATIONS_V3
from data.assembly_graph_v3 import NUM_ASM_RELATIONS, JOINT_TYPES


# ============================================================ entity encoder
class EntityEncoder(nn.Module):
    """R-GCN over the heterogeneous face+edge graph. Returns BOTH per-entity
    embeddings (for matching) and pooled per-occurrence embeddings (for the
    assembly graph). Unlike V2 it never throws the per-entity tensor away."""

    def __init__(self, in_dim=NODE_FEAT_DIM, hidden=128, out_dim=256,
                 layers=2, num_relations=NUM_RELATIONS_V3, dropout=0.1):
        super().__init__()
        self.dropout = dropout
        self.input_proj = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList(
            RGCNConv(hidden, hidden, num_relations=num_relations)
            for _ in range(layers))
        self.ent_proj = nn.Linear(hidden, out_dim)      # per-entity head
        self.pool_proj = nn.Linear(2 * hidden, out_dim)  # mean+max occ head
        self.out_dim = out_dim

    def forward(self, x_ent, edge_index, edge_type, ent_to_occ, num_occ):
        h = F.relu(self.input_proj(x_ent))
        for conv in self.convs:
            h_new = conv(h, edge_index, edge_type)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            h = h + h_new                       # residual: keep per-entity identity
        h_ent = self.ent_proj(h)                         # (M, out)
        mean = scatter(h, ent_to_occ, dim=0, dim_size=num_occ, reduce="mean")
        mx = scatter(h, ent_to_occ, dim=0, dim_size=num_occ, reduce="max")
        h_occ = self.pool_proj(torch.cat([mean, mx], dim=-1))  # (num_occ, out)
        return h_ent, h_occ


# ============================================================ assembly context
class AssemblyContext(nn.Module):
    """R-GCN over the occurrence graph. The 5 relations now INCLUDE the
    designer tree (parent>child, child>parent, sibling) on top of contact and
    kNN — the V3 ask. Tree edges are legitimate non-leakage structure."""

    def __init__(self, dim=256, layers=2, num_relations=NUM_ASM_RELATIONS,
                 dropout=0.1):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList(
            RGCNConv(dim, dim, num_relations=num_relations)
            for _ in range(layers))
        self.drops = nn.ModuleList(nn.Dropout(dropout) for _ in range(layers))

    def forward(self, h_occ, asm_edge_index, asm_edge_type):
        h = h_occ
        for i, (conv, drop) in enumerate(zip(self.convs, self.drops)):
            h = conv(h, asm_edge_index, asm_edge_type)
            if i < len(self.convs) - 1:
                h = F.relu(h)
            h = drop(h)
        return h


# ============================================================ matching head
class MatchingHead(nn.Module):
    """Cosine cross-part entity scorer with a learned temperature. Unit-norm
    projections remove the magnitude escape hatch that let embeddings collapse
    (all-similar) to trivially minimise the loss. For a jointed pair, scores
    every (entity of part i) x (entity of part j) and returns the logit matrix."""

    def __init__(self, dim=256, proj=128):
        super().__init__()
        self.proj = nn.Linear(dim, proj)
        self.logit_scale = nn.Parameter(torch.tensor(2.3))   # exp -> ~10 init

    def score(self, Hi, Hj):                 # (Ei,d),(Ej,d) -> (Ei,Ej)
        Zi = F.normalize(self.proj(Hi), dim=-1)
        Zj = F.normalize(self.proj(Hj), dim=-1)
        return (Zi @ Zj.t()) * self.logit_scale.exp().clamp(max=100.0)


# ============================================================ type head (aux)
class TypeHead(nn.Module):
    """Optional symmetric 7-class head on the pooled occurrence pair. Lets V3
    also report the V2 task so the 'axis recoverable where type isn't' claim
    is measurable in one model."""

    def __init__(self, dim=256, hidden=256, n_cls=len(JOINT_TYPES), dropout=0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3 * dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_cls))

    def forward(self, h_occ, pairs):
        a, b = h_occ[pairs[0]], h_occ[pairs[1]]
        return self.mlp(torch.cat([a + b, (a - b).abs(), a * b], dim=-1))


# ============================================================ full model
class HGCANv3(nn.Module):
    def __init__(self, geo_dim=256, hidden=256, enc_layers=2, ctx_layers=2,
                 dropout=0.1, with_type_head=True):
        super().__init__()
        self.encoder = EntityEncoder(out_dim=geo_dim, hidden=128,
                                     layers=enc_layers, dropout=dropout)
        self.no_geo = nn.Parameter(torch.zeros(geo_dim))  # empty-occ fallback
        self.context = AssemblyContext(dim=geo_dim, layers=ctx_layers,
                                       dropout=dropout)
        self.ent_ctx_proj = nn.Linear(geo_dim, geo_dim)
        self.match = MatchingHead(dim=geo_dim)
        self.type_head = TypeHead(dim=geo_dim) if with_type_head else None

    def encode(self, batch):
        """Returns context-aware per-entity embeddings z_ent and pooled
        context h_occ_ctx for the whole (possibly batched) input."""
        num_occ = int(batch.num_occ.sum()) if torch.is_tensor(batch.num_occ) \
            else batch.num_occ
        h_ent, h_occ = self.encoder(batch.x_ent, batch.ent_edge_index,
                                    batch.ent_edge_type, batch.ent_to_occ,
                                    num_occ)
        if hasattr(batch, "occ_has_geom"):
            hg = batch.occ_has_geom.unsqueeze(-1)
            h_occ = torch.where(hg, h_occ, self.no_geo.unsqueeze(0).expand_as(h_occ))
        h_occ_ctx = self.context(h_occ, batch.asm_edge_index, batch.asm_edge_type)
        # inject assembly context into every entity via its occurrence
        z_ent = self.ent_proj_add(h_ent, h_occ_ctx, batch.ent_to_occ)
        return z_ent, h_occ_ctx

    def ent_proj_add(self, h_ent, h_occ_ctx, ent_to_occ):
        return self.ent_ctx_proj(h_ent) + h_occ_ctx[ent_to_occ]

    def match_pairs(self, z_ent, batch):
        """For each jointed pair, build the entity-match logit matrix.
        Returns a list[P] of (Ei, Ej) tensors plus the per-pair global entity
        index ranges, so the loss/inference can map argmax back to entities."""
        pairs = batch.joint_occ_pairs
        ent_to_occ = batch.ent_to_occ
        mats, idx_i, idx_j = [], [], []
        for p in range(pairs.size(1)):
            oi, oj = int(pairs[0, p]), int(pairs[1, p])
            ei = (ent_to_occ == oi).nonzero(as_tuple=True)[0]
            ej = (ent_to_occ == oj).nonzero(as_tuple=True)[0]
            mats.append(self.match.score(z_ent[ei], z_ent[ej]))
            idx_i.append(ei); idx_j.append(ej)
        return mats, idx_i, idx_j

    def forward(self, batch):
        z_ent, h_occ_ctx = self.encode(batch)
        mats, idx_i, idx_j = self.match_pairs(z_ent, batch)
        type_logits = (self.type_head(h_occ_ctx, batch.joint_occ_pairs)
                       if self.type_head is not None else None)
        return mats, idx_i, idx_j, type_logits


# ============================================================ axis readout
@torch.no_grad()
def derive_axis(entity_axis, ent_global_idx):
    """Read [origin|dir] off a matched entity. ent_global_idx is the index into
    the assembly's entity_axis table. Returns (6,) or None if axis invalid."""
    row = entity_axis[ent_global_idx]
    d = row[3:]
    if torch.linalg.norm(d) < 1e-6:
        return None
    return row
