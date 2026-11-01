"""Two-tower model: item tower (content) vs query+user tower, cosine score.

  * Item tower is content-based: term bags (name / brand / category) and brand/shop/category ids. No item-id
    embedding: ~80% of displayed items never occur in any history. Popularity is NOT in the tower — it is added
    to the training score as an explicit offset (`TwoTower.pop_logit`), which forces the tower to learn what
    popularity does not already explain and leaves the exported cosine score complementary to the ranker.
  * Query+user tower: the test query shares the item tower's term embedding. Attention is candidate-aware
    (DIN-style): every candidate in the slate attends over the user's short-window history, so the user vector
    differs per candidate instead of being pooled once. Exact-match flags (same product / same top category per
    type window) enter as raw features. The mid/long windows keep no attention — their content is summarised by
    per-type counts instead (see `ctx[seg]["tcount"]`).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import C_BRAND, C_CATE, C_ID0, C_NAME, C_POP0, N_COUNTS, N_TIME_BUCKETS, SEGMENTS

CATE1 = C_ID0 + 2                       # cate_id_1 column in the item feature row
MATCH_SEGS = ("short", "mid", "long")   # windows used for the exact-match flags
N_FLAGS = 2 * len(MATCH_SEGS)           # (same product, same top category) per window


def masked_mean(e: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    m = m.unsqueeze(-1).to(e.dtype)
    return (e * m).sum(-2) / m.sum(-2).clamp(min=1.0)


class ItemTower(nn.Module):
    """Content-only item encoder: mean term embeddings of name / brand / category + id embeddings, through an MLP.

    No item-id embedding, so items that never occur in any history are encoded the same way as logged ones.
    """

    def __init__(self, n_terms: int, id_sizes: list[int], tc: dict):
        super().__init__()
        d_term, d_id, d = tc["d_term"], tc["d_id"], tc["d"]
        self.term = nn.Embedding(n_terms, d_term, padding_idx=0)
        self.ids = nn.ModuleList([nn.Embedding(n, d_id, padding_idx=0) for n in id_sizes])
        d_in = 3 * d_term + len(id_sizes) * d_id
        self.mlp = nn.Sequential(nn.Linear(d_in, 2 * d), nn.GELU(), nn.Dropout(tc.get("dropout", 0.0)),
                                 nn.Linear(2 * d, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:             # x: [..., N_FEAT] int
        x = x.long()
        parts = [masked_mean(self.term(x[..., sl]), x[..., sl] > 0) for sl in (C_NAME, C_BRAND, C_CATE)]
        parts += [emb(x[..., C_ID0 + j]) for j, emb in enumerate(self.ids)]
        return self.mlp(torch.cat(parts, -1))


class CandidateAttention(nn.Module):
    """DIN-style attention: each candidate attends over one history segment, with an always-present null slot.

    The attention score is conditioned on both the candidate and the test query, so a candidate the user's
    history speaks to pulls the matching events forward.
    """

    def __init__(self, d: int):
        super().__init__()
        self.null = nn.Parameter(torch.zeros(1, 1, d))
        self.w = nn.Linear(d, d, bias=False)
        self.q = nn.Linear(d, d, bias=False)
        self.last_attn = None

    def forward(self, cand: torch.Tensor, q: torch.Tensor, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # cand [B,S,D], q [B,D], h [B,L,D], mask [B,L] -> [B,S,D]
        B = h.size(0)
        h = torch.cat([self.null.expand(B, 1, -1).to(h.dtype), h], 1)
        mask = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=h.device), mask], 1)
        s = torch.einsum("bsd,bld->bsl", self.w(cand) + self.q(q)[:, None, :], h) / math.sqrt(h.size(-1))
        a = torch.softmax(s.float().masked_fill(~mask[:, None, :], -1e4), -1).to(h.dtype)
        self.last_attn = a.detach()
        return torch.einsum("bsl,bld->bsd", a, h)


class QueryUserTower(nn.Module):
    """Encodes (test query, user history) into one vector per candidate.

    Query terms share the item tower's embedding, so query and item text live in one space. Inputs to the output MLP:
    the query, the candidate-aware pooling of the short window, their product, the per-type counts of all three
    windows, and the exact-match flags. `use_history` / `use_query` = False zero those inputs for the ablations.
    """

    def __init__(self, item_tower: ItemTower, tc: dict, use_history: bool = True, use_query: bool = True):
        super().__init__()
        d, d_term = tc["d"], tc["d_term"]
        self.it = item_tower
        self.use_history, self.use_query = use_history, use_query
        self.q_proj = nn.Sequential(nn.Linear(d_term, d), nn.GELU(), nn.Linear(d, d))
        self.hq_proj = nn.Linear(d_term, d)                          # query under which a history event happened
        self.type_emb = nn.Embedding(5, d, padding_idx=0)
        self.time_emb = nn.Embedding(N_TIME_BUCKETS, d, padding_idx=0)
        self.attn = CandidateAttention(d)
        self.count = nn.Linear(N_COUNTS * len(SEGMENTS), d)          # per-type counts replace mid/long attention
        d_in = 4 * d + N_FLAGS
        self.out = nn.Sequential(nn.Linear(d_in, 2 * d), nn.GELU(), nn.Dropout(tc.get("dropout", 0.0)),
                                 nn.Linear(2 * d, d))

    def encode_query(self, q_terms: torch.Tensor) -> torch.Tensor:
        return self.q_proj(masked_mean(self.it.term(q_terms.long()), q_terms > 0))

    def forward(self, q_terms: torch.Tensor, ctx: dict, item_feats, cand: torch.Tensor,
                cand_rows: torch.Tensor | None = None, cand_v: torch.Tensor | None = None) -> torch.Tensor:
        """q_terms [B,Tq]; cand [B,S] item row indices.
        ctx[segment] = {item [B,L], type, time, query [B,L,Tq], mask [B,L], count [B], tcount [B,5] (4 type counts + log size)}.
        Returns [B,S,D] — one user vector per candidate."""
        q = self.encode_query(q_terms)
        if not self.use_query:
            q = torch.zeros_like(q)
        # the caller usually already has these; recomputing the item tower per step is the single biggest cost
        cand_rows = item_feats(cand) if cand_rows is None else cand_rows     # [B,S,N_FEAT]
        cand_v = self.it(cand_rows) if cand_v is None else cand_v           # [B,S,D]
        hist_rows = {s: item_feats(ctx[s]["item"]) for s in MATCH_SEGS}
        if self.use_history:
            c = ctx["short"]
            hq = c["query"].long()
            h = (self.it(hist_rows["short"]) + self.type_emb(c["type"].long())
                 + self.time_emb(c["time"].long())
                 + self.hq_proj(masked_mean(self.it.term(hq), hq > 0)))
            pooled = self.attn(cand_v, q, h, c["mask"])                # [B,S,D]
        else:
            pooled = torch.zeros_like(cand_v)
        counts = torch.cat([ctx[s]["tcount"] for s in SEGMENTS], -1).to(q.dtype)   # [B, 5*3]
        flags = self.match_flags(cand_rows, cand, hist_rows, ctx)      # [B,S,N_FLAGS]
        if not self.use_history:
            counts, flags = torch.zeros_like(counts), torch.zeros_like(flags)
        qe = q[:, None, :].expand_as(pooled)
        # qe * pooled is an explicit query-history interaction, so the MLP need not learn the product from the concatenation
        z = torch.cat([qe, pooled, qe * pooled, self.count(counts)[:, None, :].expand_as(pooled), flags], -1)
        return self.out(z)

    def match_flags(self, cand_rows: torch.Tensor, cand: torch.Tensor, hist_rows: dict,
                    ctx: dict) -> torch.Tensor:
        """Exact-match flags per window: same product, same top category as something in that history.

        History padding is row 0 and candidates are real rows (>= 1), so padding never matches."""
        cand_cate = cand_rows[..., CATE1]                              # [B,S]
        out = []
        for s in MATCH_SEGS:
            out.append((ctx[s]["item"][:, None, :] == cand[:, :, None]).any(-1).to(cand_rows.dtype))
            out.append((hist_rows[s][..., CATE1][:, None, :] == cand_cate[:, :, None]).any(-1).to(cand_rows.dtype))
        return torch.stack(out, -1)                                    # [B,S,2*len(MATCH_SEGS)]


class TwoTower(nn.Module):
    """Both towers plus the learned training-only terms: temperature, popularity offset and logQ correction.

    The exported score is the plain cosine of `user_vec` and `item_vec`; `pop_logit` and `logq` only shape the losses.
    """

    def __init__(self, n_terms: int, id_sizes: list[int], tc: dict, use_history: bool = True, use_query: bool = True):
        super().__init__()
        self.item = ItemTower(n_terms, id_sizes, tc)
        self.user = QueryUserTower(self.item, tc, use_history=use_history, use_query=use_query)
        self.log_scale = nn.Parameter(torch.tensor(math.log(20.0)))   # 1 / temperature
        self.pop_w = nn.Parameter(torch.zeros(len(("click", "cart", "order"))))   # popularity offset weights
        self.logq_coef = nn.Parameter(torch.tensor(-4.0))                         # sampled-softmax bias correction

    def item_vec(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.item(x).float(), dim=-1)

    def user_vec(self, q_terms, ctx, item_feats, cand, cand_rows=None, cand_v=None) -> torch.Tensor:
        return F.normalize(self.user(q_terms, ctx, item_feats, cand, cand_rows, cand_v).float(), dim=-1)

    def scale(self) -> torch.Tensor:
        return self.log_scale.exp().clamp(max=100.0)

    def pop_logit(self, feat_rows: torch.Tensor) -> torch.Tensor:
        """log2(1 + windowed count) summed over the three popularity windows, weighted. Bucket 0/1 -> 0."""
        return ((feat_rows[..., C_POP0:].float() - 1.0) * self.pop_w).sum(-1)

    def logq(self, feat_rows: torch.Tensor) -> torch.Tensor:
        """Share of the sampling probability explained by popularity, for the auxiliary sampled softmax."""
        return F.softplus(self.logq_coef) * (feat_rows[..., C_POP0:].float() - 1.0).sum(-1)
