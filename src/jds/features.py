"""Stage 2: hand-crafted ranking features for every displayed (user, item) pair.

Distinct roles of the data fields (the formulation behind every feature group):
  * test query + displayed slate + labels -> the supervised task (query-conditioned re-ranking of the slate);
  * the user's own history                -> personal features, always counted per event type inside that type's
                                             retention window (see windows.py), plus days since the last event;
  * query-tagged history of all users     -> query -> item knowledge (which items people engage with for a query);
  * history of all users                  -> item / attribute popularity, again per type and window;
  * product metadata                      -> content (lexical match, attributes) for the ~80% of displayed items
                                             that never appear in any history;
  * labels of *other* training users       -> impression CTR, computed out-of-fold for training users.

Leakage rules: history strictly precedes each user's test query; labels enter only through the out-of-fold CTR
tables; the shuffled slot position is never used.
Output: <scope>/features/part_XXX.parquet (one file per user chunk) and feature_list.json.
"""
from __future__ import annotations

import time

import polars as pl

from .config import check_fresh, get_logger, paths, save_json
from .prepare import check_folds, load
from .windows import CART, ORD, feature_windows, window_expr

ATTR_KEYS = ["brand_id", "shop_id", "cate_id_4"]          # attributes a user / query can have affinity for
POP_WINDOWS = ("click_7", "cart_90", "order_365")         # windows used for attribute and query->item popularity
NO_EVENT_DAYS = 999.0                                    # "days since last event" when there is none


def item_text_expr() -> pl.Expr:
    """Bag of term ids describing an item: name + brand name + 4 category names."""
    return pl.concat_list(pl.col("name_terms").fill_null([]), pl.col("brand_terms").fill_null([]),
                          *[pl.col(f"cate_terms_{i}").fill_null([]) for i in range(1, 5)]).alias("text")


def _win(cfg, names=None):
    return [w for w in feature_windows(cfg) if names is None or w[0] in names]


def _counts(df: pl.DataFrame, by: list[str], wins, prefix: str) -> pl.DataFrame:
    """Per group: number of events of each window's type inside that window."""
    return df.group_by(by).agg([window_expr(t, d).sum().cast(pl.Float32).alias(f"{prefix}_{n}") for n, t, d in wins])


# ---------------------------------------------------------------- global tables (no labels)
def popularity_tables(history: pl.DataFrame, attrs: pl.DataFrame, cfg) -> dict[str, pl.DataFrame]:
    """Item, attribute and query->item popularity from the history of all users, per type window."""
    out = {"wid": _counts(history, ["wid"], _win(cfg), "ipop").join(
        history.group_by("wid").agg(ipop_users=pl.col("user_id").n_unique().cast(pl.Float32)), on="wid")}
    h = history.join(attrs, on="wid", how="left")
    for k in ATTR_KEYS:
        out[k] = _counts(h.drop_nulls(k), [k], _win(cfg, POP_WINDOWS), f"apop_{k}")
    hq = h.filter(pl.col("qid").is_not_null())
    for k in ("wid", "brand_id", "cate_id_4"):           # query -> item / brand / category engagement of all users
        out[f"q_{k}"] = _counts(hq.drop_nulls(k), ["qid", k], _win(cfg, POP_WINDOWS), f"qpop_{k}")
    return out


# ---------------------------------------------------------------- label-based tables (out-of-fold)
CTR_KEYS = [("wid",), ("brand_id",), ("shop_id",), ("cate_id_4",), ("qid", "wid"), ("qid", "cate_id_4"), ("qid", "brand_id")]


def ctr_tables(source: pl.DataFrame, prior: float) -> dict[tuple, pl.DataFrame]:
    """Smoothed positive rate and display rate per key from labelled impressions of `source` users.

    ctr = (positives + prior * global_rate) / (impressions + prior); gain = summed label / (impressions + prior), the
    graded version, shrunk towards 0 rather than the global rate; the display rate (impressions per 1,000 source
    users) is scale-free, so tables built from different numbers of users (folds vs all training users) agree.
    """
    g = float((source["label"] > 0).mean())
    n_users = max(source["user_id"].n_unique(), 1)
    out = {}
    for key in CTR_KEYS:
        name = "_".join("q" if k == "qid" else k for k in key)
        out[key] = (source.drop_nulls(list(key)).group_by(list(key))
                    .agg(imp=pl.len(), pos=(pl.col("label") > 0).sum(), graded=pl.col("label").cast(pl.Float32).sum())
                    .select(*key,
                            ((pl.col("pos") + prior * g) / (pl.col("imp") + prior)).cast(pl.Float32).alias(f"ctr_{name}"),
                            (pl.col("graded") / (pl.col("imp") + prior)).cast(pl.Float32).alias(f"gain_{name}"),
                            (pl.col("imp") * 1000.0 / n_users).cast(pl.Float32).alias(f"disp_{name}")))
    return out


def oof_ctr_tables(labelled: pl.DataFrame, users: pl.DataFrame, cfg) -> tuple[dict, float]:
    """CTR tables per fold: fold f (training users of fold f) is built from the *other* folds' training users;
    fold -1 (validation and test users) from all training users. `labelled` = candidates with qid and attributes."""
    check_folds(users, cfg)
    tr = users.filter(pl.col("split") == "train").select("user_id", "fold")
    lab = labelled.join(tr, on="user_id")
    prior = cfg["features"]["ctr_prior"]
    tables = {-1: ctr_tables(lab, prior)}
    for f in range(cfg["prepare"]["n_folds"]):
        tables[f] = ctr_tables(lab.filter(pl.col("fold") != f), prior)
    return tables, float((lab["label"] > 0).mean())


def join_ctr(rows: pl.DataFrame, tables: dict, g_all: float) -> pl.DataFrame:
    """Each row gets the CTR features of its own fold's table (no user ever sees its own labels)."""
    parts = []
    for f, tabs in tables.items():
        r = rows.filter(pl.col("fold") == f)
        if r.is_empty():
            continue
        for key, t in tabs.items():
            r = r.join(t, on=list(key), how="left")
        parts.append(r)
    out = pl.concat(parts, how="diagonal_relaxed")
    return out.with_columns([pl.col(c).fill_null(g_all if c.startswith("ctr_") else 0.0)
                             for c in out.columns if c.startswith(("ctr_", "gain_", "disp_"))])


# ---------------------------------------------------------------- per-user features
def personal_features(rows: pl.DataFrame, hist: pl.DataFrame, cfg) -> pl.DataFrame:
    """The user's own history matched to the candidate, per key and per type window, plus days since last event."""
    wins = _win(cfg)
    out = rows
    for k in ["wid", *ATTR_KEYS]:
        ws = wins if k == "wid" else _win(cfg, POP_WINDOWS)
        agg = _counts(hist.drop_nulls(k), ["user_id", k], ws, f"u_{k}").join(
            hist.drop_nulls(k).group_by("user_id", k).agg(pl.col("age_days").min().alias(f"u_{k}_last_days")),
            on=["user_id", k])
        out = out.join(agg, on=["user_id", k], how="left")
    # carted (inside the cart window) and not ordered since the most recent cart: an open purchase intention
    w = cfg["windows"]
    co = (hist.group_by("user_id", "wid").agg(
        last_cart=pl.col("age_days").filter(window_expr(CART, w["cart"])).min(),
        last_ord=pl.col("age_days").filter(pl.col("type") == ORD).min())
        .select("user_id", "wid", u_cart_not_ordered=(pl.col("last_cart").is_not_null()
                                                      & (pl.col("last_ord").is_null() | (pl.col("last_ord") > pl.col("last_cart"))))
                .cast(pl.Float32)))
    out = out.join(co, on=["user_id", "wid"], how="left")
    return out


def query_history_features(rows: pl.DataFrame, hist: pl.DataFrame, uq: pl.DataFrame, queries: pl.DataFrame, cfg) -> pl.DataFrame:
    """Engagement under the user's earlier queries that resemble the test query (term Jaccard >= threshold).

    The exact test query practically never occurs in the user's own history (1 user in 173,831: the dataset excludes
    it), so similarity rather than identity is used.
    """
    thr = cfg["features"]["query_history_jaccard"]
    hq = (hist.filter(pl.col("qid").is_not_null()).select("user_id", "qid").unique()
          .join(queries.select("qid", hterms="terms"), on="qid")
          .join(uq.select("user_id", tterms="terms"), on="user_id")
          .with_columns(jac=(pl.col("hterms").list.set_intersection("tterms").list.len()
                             / pl.col("hterms").list.set_union("tterms").list.len().clip(1)).cast(pl.Float32))
          .select("user_id", "qid", "jac", sim=pl.col("jac") >= thr))
    h = hist.join(hq, on=["user_id", "qid"], how="inner")
    user_level = (hq.group_by("user_id").agg(u_q_sim_n=pl.col("sim").sum().cast(pl.Float32), u_q_max_jac=pl.col("jac").max())
                  .join(h.filter(pl.col("sim")).group_by("user_id").agg(u_q_sim_last_days=pl.col("age_days").min()),
                        on="user_id", how="left"))
    out = rows.join(user_level, on="user_id", how="left")
    for k in ("wid", "cate_id_4", "brand_id"):
        agg = h.filter(pl.col("sim")).drop_nulls(k).group_by("user_id", k).agg(
            pl.len().cast(pl.Float32).alias(f"u_simq_{k}_n"),
            (pl.col("type") >= CART).sum().cast(pl.Float32).alias(f"u_simq_{k}_deep"))
        out = out.join(agg, on=["user_id", k], how="left")
    return out


def user_activity(users: pl.DataFrame, hist: pl.DataFrame, cfg) -> pl.DataFrame:
    """User-level context (constant inside a slate; lets the ranker condition on how much history there is)."""
    act = _counts(hist, ["user_id"], _win(cfg), "ua")
    return (users.select("user_id", ua_hist_len=pl.col("hist_len").cast(pl.Float32),
                         ua_hist_query_share=(pl.col("n_hist_query") / pl.col("hist_len").clip(1)).cast(pl.Float32),
                         ua_gap_days=(pl.col("gap_to_query") / 86400).cast(pl.Float32))
            .join(act, on="user_id", how="left")
            .join(hist.group_by("user_id").agg(ua_first_age_days=pl.col("age_days").max()), on="user_id", how="left"))


# ---------------------------------------------------------------- lexical
def term_idf(products: pl.LazyFrame, n_docs: int) -> tuple[pl.DataFrame, float]:
    """BM25 idf over item text, estimated on a systematic sample of items."""
    n_total = products.select(pl.len()).collect().item()
    p = products.select(item_text_expr()).gather_every(max(1, n_total // int(n_docs))).collect()
    avgdl = float(p["text"].list.len().mean())
    df = p.select(pl.col("text").list.unique()).explode("text").drop_nulls().group_by("text").len()
    idf = df.select(pl.col("text").alias("term"),
                    (((p.height - pl.col("len") + 0.5) / (pl.col("len") + 0.5)) + 1.0).log().cast(pl.Float32).alias("idf"))
    return idf, avgdl


def lexical(rows: pl.DataFrame, uq: pl.DataFrame, text: pl.DataFrame, idf: pl.DataFrame, avgdl: float, cfg) -> pl.DataFrame:
    """BM25(test query, item text), query-term coverage of the full text and of the name alone."""
    k1, b = cfg["features"]["bm25"]["k1"], cfg["features"]["bm25"]["b"]
    idf_default = float(idf["idf"].max())
    items = text.with_columns(dl=pl.col("text").list.len())
    tf = items.select("wid", "text").explode("text").group_by("wid", "text").len().rename({"text": "term", "len": "tf"})
    name = items.select("wid", "name_terms").explode("name_terms").unique().rename({"name_terms": "term"}).with_columns(in_name=pl.lit(1.0))
    qt = (rows.select("user_id", "wid").join(uq.select("user_id", "terms"), on="user_id")
          .with_columns(pl.col("terms").list.unique()).explode("terms").rename({"terms": "term"}).drop_nulls("term"))
    s = (qt.join(tf, on=["wid", "term"], how="left").join(name, on=["wid", "term"], how="left")
         .join(items.select("wid", "dl"), on="wid", how="left").join(idf, on="term", how="left")
         .with_columns(tf=pl.col("tf").fill_null(0).cast(pl.Float32), idf=pl.col("idf").fill_null(idf_default),
                       dl=pl.col("dl").fill_null(avgdl).cast(pl.Float32), in_name=pl.col("in_name").fill_null(0.0))
         .with_columns(part=pl.col("idf") * pl.col("tf") * (k1 + 1) / (pl.col("tf") + k1 * (1 - b + b * pl.col("dl") / avgdl)))
         .group_by("user_id", "wid")
         .agg(lex_bm25=pl.col("part").sum(), lex_cover=(pl.col("tf") > 0).mean().cast(pl.Float32),
              lex_cover_idf=((pl.col("tf") > 0) * pl.col("idf")).sum() / pl.col("idf").sum(),
              lex_name_cover=pl.col("in_name").mean().cast(pl.Float32)))
    return rows.join(s, on=["user_id", "wid"], how="left")


# ---------------------------------------------------------------- slate context
def list_context(df: pl.DataFrame, pct_cols: list[str]) -> pl.DataFrame:
    """How typical the candidate is of its slate (proxy for the query intent) and within-slate percentile ranks."""
    n = pl.len().over("user_id")
    df = df.with_columns(
        [(pl.len().over("user_id", k) / n).cast(pl.Float32).alias(f"slate_share_{k}") for k in ATTR_KEYS]
        + [n.cast(pl.Float32).alias("slate_len")])
    return df.with_columns([((pl.col(c).rank("average").over("user_id") - 1) / (n - 1).clip(1)).cast(pl.Float32).alias(f"{c}_pct")
                            for c in pct_cols])


PCT_COLS = ["lex_bm25", "ctr_wid", "ctr_q_wid", "ipop_click_7", "ipop_order_365", "qpop_wid_click_7", "u_cate_id_4_click_7"]


# ---------------------------------------------------------------- driver
def build(cfg: dict, force: bool = False) -> list[str]:
    """Build the feature table: global tables once from all users, then per-user features one chunk of users at a time.

    Chunking bounds memory (`features.chunk_users`). Returns the feature names.
    """
    p = paths(cfg)
    log = get_logger("features", p)
    if (p.features / "feature_list.json").exists() and not force:
        check_folds(load(p.data, "users", ["split", "fold"]), cfg)
        check_fresh(p.features / "feature_list.json", p.data / "users.parquet")
        log.info(f"features exist in {p.features} (use --force to rebuild)")
        from .config import load_json
        return load_json(p.features / "feature_list.json")
    for old in p.features.glob("part_*.parquet"):
        old.unlink()
    t0 = time.time()
    users = load(p.data, "users")
    cands = load(p.data, "candidates", ["user_id", "wid", "label"])
    history = load(p.data, "history", ["user_id", "wid", "qid", "type", "age_days"])
    queries = load(p.data, "queries")
    prods = pl.scan_parquet(p.data / "products.parquet")
    attrs = prods.select("wid", *ATTR_KEYS).collect()
    uq = users.select("user_id", "qid").join(queries.select("qid", "terms"), on="qid")

    pop = popularity_tables(history, attrs, cfg)
    labelled = cands.join(users.select("user_id", "qid"), on="user_id").join(attrs, on="wid", how="left")
    ctr, g_all = oof_ctr_tables(labelled, users, cfg)
    del labelled
    idf, avgdl = term_idf(prods, cfg["features"]["bm25"]["idf_docs"])
    log.info(f"global tables in {time.time()-t0:.0f}s")

    ids = users["user_id"].to_numpy()
    step = cfg["features"]["chunk_users"]
    names = None
    for i, s in enumerate(range(0, len(ids), step)):
        chunk = pl.DataFrame({"user_id": ids[s:s + step]})
        rows = (cands.join(chunk, on="user_id", how="semi")
                .join(users.select("user_id", "qid", "split", "fold"), on="user_id")
                .join(attrs, on="wid", how="left"))
        hist = history.join(chunk, on="user_id", how="semi").join(attrs, on="wid", how="left")
        # the only label-based step: each row takes the table of its own fold, so it never sees its own label
        r = join_ctr(rows, ctr, g_all)
        r = r.join(pop["wid"], on="wid", how="left")
        for k in ATTR_KEYS:
            r = r.join(pop[k], on=k, how="left")
        for k in ("wid", "brand_id", "cate_id_4"):
            r = r.join(pop[f"q_{k}"], on=["qid", k], how="left")
        r = personal_features(r, hist, cfg)
        r = query_history_features(r, hist, uq.join(chunk, on="user_id", how="semi"), queries, cfg)
        r = r.join(user_activity(users.join(chunk, on="user_id", how="semi"), hist, cfg), on="user_id", how="left")
        text = (prods.join(rows.select("wid").unique().lazy(), on="wid", how="semi")
                .select("wid", item_text_expr(), "name_terms").collect())
        r = lexical(r, uq, text, idf, avgdl, cfg)
        r = r.join(text.select("wid", item_has_meta=pl.lit(1.0, pl.Float32), item_name_len=pl.col("name_terms").list.len().cast(pl.Float32)),
                   on="wid", how="left")
        # missing counts are zeros; missing "days since" means never
        r = r.with_columns([pl.col(c).fill_null(NO_EVENT_DAYS) for c in r.columns if c.endswith("_last_days")])
        # identifiers, the label and the raw attribute ids are never features; list_context runs last because its
        # within-slate percentiles need the final, null-filled columns
        meta = {"user_id", "wid", "label", "qid", "split", "fold", *ATTR_KEYS}
        feats = [c for c in r.columns if c not in meta]
        r = r.with_columns([pl.col(c).cast(pl.Float32).fill_null(0.0) for c in feats])
        r = list_context(r, [c for c in PCT_COLS if c in feats])
        names = names or [c for c in r.columns if c not in meta]
        r.select("user_id", "wid", "label", "split", "fold", *names).sort("user_id", "wid").write_parquet(p.features / f"part_{i:03d}.parquet")
        log.info(f"chunk {i}: {chunk.height:,} users, {r.height:,} rows, {len(names)} features ({time.time()-t0:.0f}s)")
    save_json(names, p.features / "feature_list.json")
    return names


def read(p, columns=None, splits=None, user_frac: float = 1.0, seed: int = 0) -> pl.DataFrame:
    """Load the feature table (optionally only some splits and a deterministic share of users)."""
    lf = pl.scan_parquet(p.features / "part_*.parquet")
    if splits:
        lf = lf.filter(pl.col("split").is_in(list(splits)))
    if user_frac < 1.0:
        lf = lf.filter((pl.col("user_id").hash(seed) % 10_000) < int(user_frac * 10_000))
    if columns:
        lf = lf.select(columns)
    return lf.collect()
