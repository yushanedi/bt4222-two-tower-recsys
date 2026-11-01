"""Stage 2c: the previous team's Hybrid NCF + LSTM model, re-implemented as a reference baseline.

Their setup: per-candidate 4-class classification (label 0-3) trained on lists subsampled to 1 positive : 4 negatives,
user = 2-layer LSTM + attention over the last 200 history events (item id + type + time-gap bucket) plus 11 user
statistics, candidate = item id + brand/shop/category embeddings. Two variants:
  * `published`: their attribute lookup indexed the product table with the *history* item index (a bug), ranked by
    the max class probability, as published;
  * `fixed`: the product's own attribute row, ranked by the expected label.
Both are evaluated on the full displayed slates of test users (the protocol every other model here uses), and on
their own sampled 1:4 lists for reference.
"""
from __future__ import annotations

import time

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .config import get_device, get_logger, paths, save_json, set_seed
from .metrics import evaluate, per_user_metrics, results_table
from .prepare import load

ATTR_COLS = ["brand_id", "shop_id", "cate_id_1", "cate_id_2", "cate_id_3", "cate_id_4"]


class Attention(nn.Module):
    """Additive attention pooling of LSTM outputs; it ignores the candidate and the query."""

    def __init__(self, d):
        super().__init__()
        self.attn = nn.Linear(d, 1)

    def forward(self, x, mask):
        s = self.attn(x).squeeze(-1).masked_fill(~mask, float("-inf"))
        return (x * torch.softmax(s, 1).unsqueeze(-1)).sum(1)


class HybridNCF(nn.Module):
    """MLP over (user sequence, candidate, user statistics) with 4 output classes, one per label 0-3.

    The candidate is its item-id embedding plus attribute embeddings found through `lookups`, indexed by the
    variant's product index. The query is not an input, so the model cannot condition on what the user searched for.
    """

    def __init__(self, n_items, n_attr, n_meta, lookups, d=64, hidden=128):
        super().__init__()
        self.item_emb = nn.Embedding(n_items, d, padding_idx=0)
        self.type_emb = nn.Embedding(6, d, padding_idx=0)
        self.time_emb = nn.Embedding(12, d, padding_idx=0)
        self.lstm = nn.LSTM(d, d, num_layers=2, batch_first=True, dropout=0.3)
        self.attn = Attention(d)
        self.empty_user = nn.Parameter(torch.zeros(1, d))
        self.cand_wid = nn.Embedding(n_items, d, padding_idx=0)
        self.attr_emb = nn.ModuleList([nn.Embedding(n, d, padding_idx=0) for n in n_attr])
        self.register_buffer("lookups", lookups)
        self.meta = nn.Linear(n_meta, d)
        self.mlp = nn.Sequential(nn.Linear(3 * d, hidden), nn.ReLU(), nn.BatchNorm1d(hidden), nn.Dropout(0.3),
                                 nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.BatchNorm1d(hidden // 2), nn.Dropout(0.3),
                                 nn.Linear(hidden // 2, 4))

    def user(self, item, typ, tim, lens):
        x = self.item_emb(item) + self.type_emb(typ) + self.time_emb(tim)
        out = self.empty_user.expand(x.size(0), -1).clone()
        ne = lens > 0
        if ne.any():
            L = int(lens[ne].max())
            packed = pack_padded_sequence(x[ne][:, :L], lens[ne].cpu(), batch_first=True, enforce_sorted=False)
            o, _ = self.lstm(packed)
            o, _ = pad_packed_sequence(o, batch_first=True, total_length=L)
            mask = torch.arange(L, device=x.device)[None] < lens[ne][:, None]
            out[ne] = self.attn(o, mask)
        return out

    def forward(self, item, typ, tim, lens, meta, c_hidx, c_attr, c_user):
        """c_* are flat candidate tensors; c_user maps each candidate to its row in the user batch."""
        u = self.user(item, typ, tim, lens)[c_user]
        a = self.lookups[:, c_attr]
        c = self.cand_wid(c_hidx) + sum(e(a[j]) for j, e in enumerate(self.attr_emb))
        m = self.meta(meta)[c_user]
        return self.mlp(torch.cat([u, c, m], -1))


def run(cfg: dict) -> pl.DataFrame:
    """Rebuild the previous team's data pipeline, diagnose its gaps, then train and evaluate each variant."""
    p = paths(cfg)
    log = get_logger("prior_ncf", p)
    nc = cfg["prior_ncf"]
    device = get_device(cfg)
    seed = cfg["seed"]
    set_seed(seed)
    users = load(p.data, "users")
    cands = load(p.data, "candidates", ["user_id", "wid", "label"])
    hist = load(p.data, "history", ["user_id", "step", "wid", "type", "dt"])
    products = load(p.data, "products", ["wid", *ATTR_COLS])

    # their training data: every positive + up to 4 random negatives per positive
    rng = np.random.default_rng(seed)
    sampled = (cands.with_columns(_r=pl.Series(rng.random(cands.height)), is_pos=pl.col("label") > 0)
               .with_columns(n_pos=pl.col("is_pos").sum().over("user_id"),
                             neg_rank=pl.col("_r").rank("ordinal").over(["user_id", "is_pos"]))
               .filter(pl.col("is_pos") | (pl.col("neg_rank") <= nc["neg_per_pos"] * pl.col("n_pos")))
               .select("user_id", "wid", "label"))

    # item vocabulary from history (hidx, 0 = unseen or rare) and a separate product table with its attribute lookups (pidx)
    hv = (hist.group_by("wid").len().filter(pl.col("len") >= nc["min_item_count"]).sort("wid")
          .with_row_index("hidx", offset=1).select("wid", pl.col("hidx").cast(pl.Int64)))
    n_hist = hv.height + 1
    prod_tab = (products.join(pl.concat([cands.select("wid"), hv.select("wid")]).unique(), on="wid", how="semi")
                .with_row_index("pidx", offset=1).with_columns(pl.col("pidx").cast(pl.Int64)))
    sizes = {}
    for c in ATTR_COLS:
        u = prod_tab[c].drop_nulls().unique().sort()
        prod_tab = prod_tab.join(pl.DataFrame({c: u, f"{c}_i": np.arange(1, len(u) + 1)}), on=c, how="left")
        sizes[c] = len(u) + 1
    prod_tab = prod_tab.with_columns(pl.col([f"{c}_i" for c in ATTR_COLS]).fill_null(0)).sort("pidx")
    P = prod_tab.height + 1
    lookups = torch.zeros((len(ATTR_COLS), P), dtype=torch.long)
    for j, c in enumerate(ATTR_COLS):
        lookups[j, 1:] = torch.from_numpy(prod_tab[f"{c}_i"].to_numpy().astype(np.int64))

    # user sequences: each user's last `maxlen` events, left-aligned, with gaps cut into 10 log-spaced time bins
    L = int(nc["maxlen"])
    h = (hist.join(hv, on="wid", how="left").with_columns(pl.col("hidx").fill_null(0))
         .with_columns(rk=pl.col("step").rank("ordinal", descending=True).over("user_id"))
         .filter(pl.col("rk") <= L).sort("user_id", "step"))
    log_t = np.log1p(hist["dt"].to_numpy())
    bins = np.round(np.expm1(np.linspace(log_t.min(), log_t.max(), 10))).astype(np.int64)
    U = int(users["user_id"].max()) + 1
    seq_item, seq_type, seq_time = (np.zeros((U, L), np.int64) for _ in range(3))
    lengths = np.zeros(U, np.int64)
    uid = h["user_id"].to_numpy()
    pos = h.select(pl.int_range(pl.len()).over("user_id")).to_series().to_numpy()
    seq_item[uid, pos] = h["hidx"].to_numpy()
    seq_type[uid, pos] = h["type"].to_numpy()
    seq_time[uid, pos] = np.digitize(h["dt"].to_numpy(), bins)
    np.add.at(lengths, uid, 1)
    del h

    # user statistics over the whole history, standardised with training users' mean and std
    hc = hist.join(products.select("wid", "cate_id_1", "cate_id_2", "cate_id_3", "cate_id_4"), on="wid", how="left")
    meta = (hc.group_by("user_id").agg(
        *[pl.col(f"cate_id_{i}").n_unique().alias(f"nunique_cats_{i}") for i in range(1, 5)],
        history_len=pl.len(), click_ratio=(pl.col("type") == 1).mean(), cart_ratio=(pl.col("type") == 3).mean(),
        order_ratio=(pl.col("type") == 4).mean(), avg_gap=pl.col("dt").mean(), std_gap=pl.col("dt").std().fill_null(0),
        max_gap=pl.col("dt").max()).sort("user_id"))
    del hc
    meta_cols = [c for c in meta.columns if c != "user_id"]
    M = np.zeros((U, len(meta_cols)), np.float32)
    M[meta["user_id"].to_numpy()] = meta.select(meta_cols).to_numpy().astype(np.float32)
    tr_ids = users.filter(pl.col("split") == "train")["user_id"].to_numpy()
    M = (M - M[tr_ids].mean(0)) / (M[tr_ids].std(0) + 1e-6)

    def group_lists(rows):
        r = (rows.join(hv, on="wid", how="left").join(prod_tab.select("wid", "pidx"), on="wid", how="left")
             .with_columns(pl.col("hidx", "pidx").fill_null(0)).sort("user_id"))
        # "published" deliberately reproduces their bug: the history index is used to look up the product table, so the
        # attributes belong to an unrelated product. "fixed" uses the item's own product index
        return {"user_id": r["user_id"].to_numpy(), "hidx": r["hidx"].to_numpy(),
                "published": r["hidx"].clip(0, P - 1).to_numpy(), "fixed": r["pidx"].to_numpy(),
                "label": r["label"].to_numpy().astype(np.int64), "wid": r["wid"].to_numpy()}

    split_users = {s: users.filter((pl.col("split") == s) & pl.col("has_pos")).select("user_id") for s in ("train", "val", "test")}
    samp = {s: group_lists(sampled.join(u, on="user_id", how="semi")) for s, u in split_users.items()}
    full_test = group_lists(cands.join(split_users["test"], on="user_id", how="semi"))

    # ---- gap diagnosis: structural reasons this model cannot do well on the task (results/prior_ncf_gaps.json)
    ft = full_test
    pos = ft["label"] > 0
    attr_pub, attr_fix = lookups[:, torch.from_numpy(ft["published"])].numpy(), lookups[:, torch.from_numpy(ft["fixed"])].numpy()
    has_meta = ft["fixed"] > 0
    wrong = (attr_pub != attr_fix).any(0)
    gaps = {
        "uses_test_query": False,
        "displayed_items_without_item_embedding (cold, not in history >= min_item_count)": float((ft["hidx"] == 0).mean()),
        "positives_without_item_embedding": float((ft["hidx"][pos] == 0).mean()),
        "published_variant_wrong_attributes_share (items with metadata)": float(wrong[has_meta].mean()) if has_meta.any() else None,
        "published_variant_wrong_attributes_share_positives": float(wrong[has_meta & pos].mean()) if (has_meta & pos).any() else None,
        "train_items_per_list_sampled": float(np.bincount(np.unique(samp["train"]["user_id"], return_inverse=True)[1]).mean()),
        "test_items_per_slate_full": float(np.bincount(np.unique(ft["user_id"], return_inverse=True)[1]).mean()),
        "history_truncated_to_last_events": L,
        "users_with_history_longer_than_maxlen": float((users["hist_len"].to_numpy() > L).mean()),
        "users_with_history": int((lengths > 0).sum()),
        "user_meta_features": meta_cols,
        "note": "click/cart/order ratios over the whole history mix retention windows (see eda.json recency_confound)",
    }
    save_json(gaps, p.results / "prior_ncf_gaps.json")
    log.info(f"prior model gap diagnosis: {gaps}")

    def batches(data, variant, bs, shuffle=False):
        # batches of users; candidates stay flat and c_user maps each one back to its user's row
        uids = np.unique(data["user_id"])
        starts = np.searchsorted(data["user_id"], uids)
        ends = np.searchsorted(data["user_id"], uids, side="right")
        order = rng.permutation(len(uids)) if shuffle else np.arange(len(uids))
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            b_u = uids[idx]
            rows = np.concatenate([np.arange(starts[i], ends[i]) for i in idx])
            c_user = np.repeat(np.arange(len(idx)), ends[idx] - starts[idx])

            def t(a):
                return torch.from_numpy(np.ascontiguousarray(a)).to(device)
            yield ((t(seq_item[b_u]), t(seq_type[b_u]), t(seq_time[b_u]), t(lengths[b_u]).clamp(max=L), t(M[b_u]),
                    t(data["hidx"][rows]), t(data[variant][rows]), t(c_user)), t(data["label"][rows]), rows)

    def train(variant):
        # inverse-frequency class weights on the sampled lists; keeps the epoch with the lowest validation loss
        set_seed(seed)
        model = HybridNCF(n_hist, [sizes[c] for c in ATTR_COLS], len(meta_cols), lookups, nc["d"], nc["hidden"]).to(device)
        cnt = np.bincount(samp["train"]["label"], minlength=4).astype(np.float32) + 1.0
        crit = nn.CrossEntropyLoss(weight=torch.tensor((1 / cnt) / (1 / cnt).sum(), device=device))
        opt = torch.optim.Adam(model.parameters(), lr=nc["lr"], weight_decay=1e-5)
        best, best_state = 1e9, None
        for ep in range(nc["epochs"]):
            model.train()
            t0, tl, n = time.time(), 0.0, 0
            for x, y, _ in batches(samp["train"], variant, nc["batch_users"], shuffle=True):
                loss = crit(model(*x), y)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tl, n = tl + loss.item(), n + 1
            model.eval()
            vl = vn = 0
            with torch.no_grad():
                for x, y, _ in batches(samp["val"], variant, 1024):
                    vl, vn = vl + crit(model(*x), y).item(), vn + 1
            log.info(f"[{variant}] epoch {ep + 1}: train loss {tl / max(n, 1):.4f} val loss {vl / max(vn, 1):.4f} ({time.time() - t0:.0f}s)")
            if vl / max(vn, 1) < best:
                best, best_state = vl / max(vn, 1), {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        return model.eval()

    @torch.no_grad()
    def score(model, data, variant):
        # both ranking scores: max class probability (their choice) and expected label (the fixed variant's)
        maxp = np.zeros(len(data["label"]), np.float32)
        expv = np.zeros_like(maxp)
        for x, _, rows in batches(data, variant, 512):
            pr = torch.softmax(model(*x).float(), -1)
            maxp[rows] = pr.max(-1).values.cpu().numpy()
            expv[rows] = (pr * torch.arange(4, device=pr.device)).sum(-1).cpu().numpy()
        return pl.DataFrame({"user_id": data["user_id"], "wid": data["wid"], "label": data["label"],
                             "score_maxprob": maxp, "score_expected": expv})

    res = []
    for variant in nc["variants"]:
        model = train(variant)
        sc = "score_maxprob" if variant == "published" else "score_expected"
        s_full = score(model, full_test, variant)
        res.append(evaluate(s_full, sc, seed=seed, name=f"Prior team Hybrid NCF+LSTM ({variant}) [test]"))
        res.append(evaluate(score(model, samp["test"], variant), sc, seed=seed,
                            name=f"Prior team Hybrid NCF+LSTM ({variant}), their sampled 1:4 lists [reference]"))
        per_user_metrics(s_full, sc, seed=seed).write_parquet(p.results / f"per_user_prior_ncf_{variant}_test.parquet")
    save_json(res, p.results / "prior_ncf.json")
    tbl = results_table(res)
    log.info("\n" + str(tbl))
    return tbl
