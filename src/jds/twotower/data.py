"""Two-tower data plumbing: vocabularies, the item feature matrix, and batch assembly by pure array indexing.

History segments (see windows.py): relative to an anchor (the test query, or a historical event for the auxiliary
task) the user's earlier events are split by age into
    short  = [0, short) days   -> every event type is recorded (clicks only exist here)
    mid    = [short, mid) days -> carts, follows and orders only
    long   = [mid, inf) days   -> orders only
and the most recent `seg_len[segment]` events of each segment are kept. Each segment therefore has a homogeneous
meaning, instead of one window whose content depends on how active the user was last week. The model attends over
the short segment only; mid and long enter through per-type counts and exact-match flags.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from ..windows import CART, CLICK, ORD, window_expr

# ---------------------------------------------------------------- item feature matrix layout
NAME_W, BRAND_W, CATE_W = 20, 4, 6
ID_COLS = ["brand_id", "shop_id", "cate_id_1", "cate_id_2", "cate_id_3", "cate_id_4"]
C_NAME = slice(0, NAME_W)
C_BRAND = slice(NAME_W, NAME_W + BRAND_W)
C_CATE = slice(NAME_W + BRAND_W, NAME_W + BRAND_W + CATE_W)
C_ID0 = NAME_W + BRAND_W + CATE_W
POP_COLS = [("click", CLICK, "click"), ("cart", CART, "cart"), ("order", ORD, "order")]   # (name, type, window key)
C_POP0 = C_ID0 + len(ID_COLS)
N_FEAT = C_POP0 + len(POP_COLS)
N_POP_BUCKETS = 18                       # 0 = pad, 1 = never, 2..17 = log2 buckets
N_TIME_BUCKETS = 24                      # 0 = pad, 1..23 = log2(seconds) buckets
N_TYPES = 4                              # CLICK, FLW, CART, ORD
N_COUNTS = N_TYPES + 1                   # per-type counts + the true log size, per segment
SEGMENTS = ("short", "mid", "long")
TYPE_WEIGHT = np.array([0.0, 1.0, 1.0, 2.0, 3.0], np.float32)   # auxiliary loss weight by type code (CLICK, FLW, CART, ORD)


def pop_bucket(x: np.ndarray) -> np.ndarray:
    return (np.minimum(16, np.floor(np.log2(1 + x))) + 1).astype(np.int32)


def time_bucket(sec: np.ndarray) -> np.ndarray:
    return (np.clip(np.floor(np.log2(1 + np.maximum(sec, 0))), 0, N_TIME_BUCKETS - 2) + 1).astype(np.int64)


# ---------------------------------------------------------------- vocabularies
class TermVocab:
    """Frequent terms get their own index, the rest are hashed into OOV buckets; 0 = padding."""

    def __init__(self, terms_sorted: np.ndarray, oov_buckets: int):
        self.terms = np.asarray(terms_sorted, dtype=np.int64)
        self.V = len(self.terms)
        self.oov = int(oov_buckets)
        self.size = self.V + 1 + self.oov

    def map(self, a: np.ndarray) -> np.ndarray:
        """int array with -1 as padding -> int32 indices."""
        out = np.zeros(a.shape, dtype=np.int32)
        m = a >= 0
        v = a[m].astype(np.int64)
        if self.V:
            pos = np.minimum(np.searchsorted(self.terms, v), self.V - 1)
            found = self.terms[pos] == v
        else:
            pos, found = np.zeros(len(v), np.int64), np.zeros(len(v), bool)
        out[m] = np.where(found, pos + 1, self.V + 1 + (v % self.oov)).astype(np.int32)
        return out


class IdVocab:
    """Ids seen at least `min_count` times get an index; rare / unseen ids hash into `buckets` slots (or map to 0)."""

    def __init__(self, values_sorted: np.ndarray, buckets: int = 0):
        self.values = np.asarray(values_sorted, dtype=np.int64)
        self.buckets = int(buckets)
        self.size = len(self.values) + 1 + self.buckets

    def map(self, a: np.ndarray) -> np.ndarray:
        out = np.zeros(a.shape, dtype=np.int32)
        if not len(self.values):
            return out
        m = a >= 0
        v = a[m].astype(np.int64)
        pos = np.minimum(np.searchsorted(self.values, v), len(self.values) - 1)
        found = self.values[pos] == v
        # Long-tail ids still carry a little signal, so hash them into shared buckets instead of one dead index.
        out[m] = np.where(found, pos + 1, len(self.values) + 1 + (v % self.buckets)) if self.buckets \
            else np.where(found, pos + 1, 0)
        return out


def build_term_vocab(products: pl.LazyFrame, queries: pl.DataFrame, tc: dict) -> TermVocab:
    """Count terms over all distinct queries + a systematic sample of product texts; keep frequent ones."""
    n_total = products.select(pl.len()).collect().item()
    p = (products.select("name_terms", "brand_terms", "cate_terms_3", "cate_terms_4")
         .gather_every(max(1, n_total // int(tc["term_count_docs"]))).collect())
    parts = [p[c].explode().drop_nulls() for c in p.columns] + [queries["terms"].explode().drop_nulls()]
    cnt = pl.concat([s.cast(pl.Int64).alias("t").to_frame() for s in parts]).group_by("t").len()
    keep = cnt.filter(pl.col("len") >= tc["min_term_count"]).sort("len", descending=True).head(int(tc["max_terms"]))
    return TermVocab(np.sort(keep["t"].to_numpy()), tc["oov_buckets"])


def build_id_vocabs(products: pl.LazyFrame, min_count: int = 2) -> dict:
    out = {}
    for c in ID_COLS:
        vc = products.select(c).drop_nulls().group_by(c).len().filter(pl.col("len") >= min_count).collect()
        out[c] = IdVocab(np.sort(vc[c].to_numpy()), buckets=2048 if c in ("brand_id", "shop_id") else 0)
    return out


def _padded(df: pl.DataFrame, col: str, w: int) -> np.ndarray:
    """List column -> [n, w] int64 matrix (truncated to w, padded with -1)."""
    s = df[col].fill_null([]).list.head(w)
    lens = s.list.len().to_numpy().astype(np.int64)
    flat = s.explode().drop_nulls().to_numpy().astype(np.int64)
    out = np.full((df.height, w), -1, dtype=np.int64)
    rows = np.repeat(np.arange(df.height), lens)
    cols = np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens)
    out[rows, cols] = flat
    return out


def item_popularity(history: pl.DataFrame, cfg) -> pl.DataFrame:
    """Per item: clicks in the click window, carts in the cart window, orders in the order window (all users)."""
    w = cfg["windows"]
    return history.group_by("wid").agg([window_expr(t, w[k]).sum().alias(f"pop_{n}") for n, t, k in POP_COLS])


def build_item_matrix(item_wids: np.ndarray, products_path, tv: TermVocab, idv: dict, pop: pl.DataFrame,
                      chunk: int = 1_000_000) -> np.ndarray:
    """Row r (1-based) describes item_wids[r-1]; row 0 is the all-zero padding item.

    Columns: 20 name terms | 4 brand terms | 6 category terms (L3 + L4) | 6 ids | 3 windowed popularity buckets.
    Items without metadata keep zero text / ids and still get popularity.
    """
    n = len(item_wids)
    X = np.zeros((n + 1, N_FEAT), dtype=np.int32)
    lf = pl.scan_parquet(products_path)
    total = lf.select(pl.len()).collect().item()
    for s in range(0, total, chunk):
        df = lf.slice(s, chunk).collect()
        w = df["wid"].to_numpy()
        rows = np.searchsorted(item_wids, w)
        ok = (rows < n) & (item_wids[np.minimum(rows, n - 1)] == w)
        df, rows = df.filter(pl.Series(ok)), rows[ok] + 1
        cate = df.select(pl.concat_list(pl.col("cate_terms_3").fill_null([]).list.head(3),
                                        pl.col("cate_terms_4").fill_null([]).list.head(3)).alias("c"))
        X[rows, C_NAME] = tv.map(_padded(df, "name_terms", NAME_W))
        X[rows, C_BRAND] = tv.map(_padded(df, "brand_terms", BRAND_W))
        X[rows, C_CATE] = tv.map(_padded(cate, "c", CATE_W))
        for j, c in enumerate(ID_COLS):
            X[rows, C_ID0 + j] = idv[c].map(df[c].fill_null(-1).to_numpy())
    X[1:, C_POP0:] = 1                                        # 1 = never interacted with
    pw = pop["wid"].to_numpy()
    r = np.searchsorted(item_wids, pw)
    ok = (r < n) & (item_wids[np.minimum(r, n - 1)] == pw)
    for j, (name, _, _) in enumerate(POP_COLS):
        X[r[ok] + 1, C_POP0 + j] = pop_bucket(pop[f"pop_{name}"].to_numpy()[ok])
    return X


# ---------------------------------------------------------------- batch assembly
class Stores:
    """Numpy views of history, queries, slates and items; every batch is assembled by indexing."""

    def __init__(self, history: pl.DataFrame, users: pl.DataFrame, cands: pl.DataFrame, qterms: np.ndarray,
                 item_wids: np.ndarray, cfg: dict):
        tc, w = cfg["tower"], cfg["windows"]
        self.seg_len = {s: int(tc["seg_len"][s]) for s in SEGMENTS}
        self.short, self.mid = float(w["short"]), float(w["mid"])
        self.max_slate = int(tc["max_slate"])
        h = history.sort("user_id", "step")
        self.n_users = int(users["user_id"].max()) + 1
        self.h_user = h["user_id"].to_numpy().astype(np.int64)
        self.h_item = (np.searchsorted(item_wids, h["wid"].to_numpy()) + 1).astype(np.int64)
        self.h_type = h["type"].to_numpy().astype(np.int64)
        self.h_age = h["age_days"].to_numpy().astype(np.float64)
        self.h_qid = h["qid"].fill_null(-1).to_numpy().astype(np.int64)
        # sortable key: user, then age decreasing (ages never increase along a user's history). Each user owns a band of
        # 4096 and ages stay below 2048 days, so one searchsorted over h_key finds age boundaries for a batch of users
        self.h_key = self.h_user * 4096.0 + (2048.0 - self.h_age)
        self.u_start = np.searchsorted(self.h_user, np.arange(self.n_users))
        self.u_end = np.searchsorted(self.h_user, np.arange(self.n_users), side="right")
        self.u_qid = np.full(self.n_users, -1, np.int64)
        self.u_qid[users["user_id"].to_numpy()] = users["qid"].to_numpy()
        self.qterms = qterms                      # [n_queries + 1, Tq] term indices, last row = empty query
        self.item_wids = item_wids
        # slates: [n_users, max_slate] item index (0 = pad) and label (-1 = pad)
        c = cands.sort("user_id", "pos")
        cu = c["user_id"].to_numpy().astype(np.int64)
        cpos = (np.arange(len(cu)) - np.searchsorted(cu, cu)).astype(np.int64)
        keep = cpos < self.max_slate
        self.s_item = np.zeros((self.n_users, self.max_slate), np.int64)
        self.s_label = np.full((self.n_users, self.max_slate), -1, np.int8)
        self.s_item[cu[keep], cpos[keep]] = np.searchsorted(item_wids, c["wid"].to_numpy()[keep]) + 1
        self.s_label[cu[keep], cpos[keep]] = c["label"].to_numpy()[keep]
        self.s_wid = np.zeros((self.n_users, self.max_slate), np.int64)
        self.s_wid[cu[keep], cpos[keep]] = c["wid"].to_numpy()[keep]

    def _bound(self, users: np.ndarray, anchor_age: np.ndarray, days: float) -> np.ndarray:
        """Index of the first event of `users` that is less than `days` older than the anchor."""
        return np.searchsorted(self.h_key, users * 4096.0 + (2048.0 - anchor_age - days), side="right")

    def context(self, users: np.ndarray, end: np.ndarray, anchor_age: np.ndarray) -> dict:
        """Three history segments of the events before `end` (exclusive), right-aligned, ages relative to the anchor.

        History is sorted oldest to newest, so each segment is one contiguous block [lo, hi): long, then mid, then short.
        """
        start = self.u_start[users]
        b_long = np.clip(self._bound(users, anchor_age, self.mid), start, end)
        b_mid = np.clip(self._bound(users, anchor_age, self.short), b_long, end)
        out = {}
        for seg, lo, hi in (("long", start, b_long), ("mid", b_long, b_mid), ("short", b_mid, end)):
            L = self.seg_len[seg]
            # the L positions ending just before hi: newest event in the last slot, positions before lo become padding
            idx = hi[:, None] - L + np.arange(L)[None]
            mask = idx >= lo[:, None]
            idx = np.where(mask, idx, 0)
            t = np.where(mask, self.h_type[idx], 0)
            out[seg] = {
                "item": np.where(mask, self.h_item[idx], 0),
                "type": np.where(mask, self.h_type[idx], 0),
                "time": np.where(mask, time_bucket((self.h_age[idx] - anchor_age[:, None]) * 86400.0), 0),
                "query": self.qterms[np.where(mask, self.h_qid[idx], -1)],      # -1 -> empty query row
                "mask": mask,
                "count": np.log1p(hi - lo).astype(np.float32),
                # shortcut: per-type counts are capped at seg_len; the exact total rides along as the last column.
                "tcount": np.concatenate([np.stack([(t == k).sum(1) for k in range(1, N_TYPES + 1)], -1),
                                          np.log1p(hi - lo)[:, None]], -1).astype(np.float32),
            }
        return out

    def user_batch(self, users: np.ndarray) -> dict:
        """Test-time anchor: the user's test query and the full history before it."""
        users = np.asarray(users, np.int64)
        return {"q": self.qterms[self.u_qid[users]],
                "ctx": self.context(users, self.u_end[users], np.zeros(len(users)))}

    def slate_batch(self, users: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray]:
        users = np.asarray(users, np.int64)
        return self.user_batch(users), self.s_item[users], self.s_label[users]

    def pair_batch(self, ev: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
        """Auxiliary pairs: (query of the event, the history before it) -> item of the event, with type weight."""
        u = self.h_user[ev]
        b = {"q": self.qterms[self.h_qid[ev]], "ctx": self.context(u, ev, self.h_age[ev])}
        return b, self.h_item[ev], TYPE_WEIGHT[self.h_type[ev]], u

    def neg_batch(self, users: np.ndarray, pos: np.ndarray, k: int,
                  allowed: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Up to k items from the anchor user's slate as hard negatives: products shown and passed over (label 0),
        excluding the anchor's own item. A far harder negative than another anchor's positive.

        The slate is the user's final (labelled) search, so only users in `allowed` (bool [n_users], the model's
        training users) may contribute: otherwise validation / test / held-out-fold slates leak into training and
        their positives are pushed down. Returns (row indices [B,k], validity mask [B,k])."""
        users = np.asarray(users, np.int64)
        sl = self.s_item[users]
        keep = (sl > 0) & (self.s_label[users] == 0) & (sl != np.asarray(pos)[:, None])
        if allowed is not None:
            keep &= allowed[users][:, None]
        order = np.argsort(~keep, axis=1, kind="stable")[:, :k]
        return np.take_along_axis(sl, order, 1), np.take_along_axis(keep, order, 1)

    def aux_events(self, max_age: float) -> np.ndarray:
        """Query-tagged history events inside the click window: anchors whose earlier history is fully recorded."""
        return np.flatnonzero((self.h_qid >= 0) & (self.h_age < max_age))


def query_matrix(queries: pl.DataFrame, tv: TermVocab, q_len: int) -> np.ndarray:
    """[max_qid + 2, q_len] term indices; row -1 (the last row) is an all-padding 'no query'."""
    n = int(queries["qid"].max()) + 1
    Q = np.zeros((n + 1, q_len), np.int32)
    Q[queries["qid"].to_numpy()] = tv.map(_padded(queries, "terms", q_len))
    return Q
