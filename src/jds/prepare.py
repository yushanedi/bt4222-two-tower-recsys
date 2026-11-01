"""Stage 0: raw JDsearch text files -> typed parquet tables, leakage-free slate order, user split and folds.

Outputs in <work>/full/data/ (and <work>/sample_<N>/data/ for subsamples):
  queries.parquet     qid, terms (list of term ids), n_terms
  users.parquet       user_id, qid (test query), n_cand, n_pos, n_click, n_cart, n_ord, max_label,
                      hist_len, n_hist_query, gap_to_query, split, has_pos, fold
  candidates.parquet  user_id, pos (shuffled slot), wid, label, orig_pos (audit only, never a feature)
  history.parquet     user_id, step, wid, qid (null = query-less), type, dt, t_before_query, age_days
  products.parquet    wid, name_terms, brand_id, brand_terms, cate_id_1..4, cate_terms_1..4, shop_id
"""
from __future__ import annotations

import math
import time

import numpy as np
import polars as pl

from .config import TERM_SEP, TYPE_CODE, get_logger, paths, sample_size, save_json

FILES = ("queries", "users", "candidates", "history", "products")


def _canon(expr: pl.Expr) -> pl.Expr:
    """Canonical query string: term ids joined by ^X with empty tokens removed."""
    return expr.str.split(TERM_SEP).list.eval(pl.element().filter(pl.element() != "")).list.join(TERM_SEP)


def _terms(col: str) -> pl.Expr:
    return pl.col(col).str.split(TERM_SEP).list.eval(pl.element().filter(pl.element() != "").cast(pl.Int32))


def done(data_dir) -> bool:
    """A stage-0 output directory is complete only once the marker written at the very end exists."""
    return (data_dir / "_DONE").exists() and all((data_dir / f"{f}.parquet").exists() for f in FILES)


def prepare_full(cfg: dict, force: bool = False) -> None:
    """Parse the raw files into the tables listed in the module docstring, after checking the raw lists line up.

    Candidates are shuffled because the raw order leaks the label (positives first). Users are split among those with
    a positive; users without one go to training, then every training user gets a cross-fitting fold.
    """
    p = paths(cfg, "full")
    log = get_logger("prepare", p)
    if done(p.data) and not force:
        sync_folds(p.data, cfg, log)
        log.info(f"full data already prepared in {p.data} (use --force to rebuild)")
        return
    user_txt, prod_txt = p.raw / "user_behavior_data.txt", p.raw / "product_meta_data.txt"
    for f in (user_txt, prod_txt):
        if not f.exists():
            raise FileNotFoundError(f"{f} not found: put the raw JDsearch files in {p.raw} (see data/raw/README.md)")
    seed = cfg["seed"]
    t0 = time.time()
    raw = pl.read_csv(user_txt, separator="\t", quote_char=None, infer_schema=False).with_row_index("user_id")
    log.info(f"user file: {raw.shape} in {time.time()-t0:.0f}s")

    # ---- integrity checks documented in the dataset README
    lens = raw.select(*[pl.col(c).str.split("_").list.len().alias(c) for c in
                        ("history_qry_list", "history_wid_list", "history_type_list", "history_time_list",
                         "candidate_wid_list", "candidate_label_list")])
    checks = {
        "history lists have equal length": bool(((lens["history_qry_list"] == lens["history_wid_list"])
                                                 & (lens["history_wid_list"] == lens["history_type_list"])).all()),
        "time list is one longer (includes the test query)": bool((lens["history_time_list"] == lens["history_wid_list"] + 1).all()),
        "candidate ids and labels have equal length": bool((lens["candidate_wid_list"] == lens["candidate_label_list"]).all()),
    }
    if not all(checks.values()):
        raise ValueError(f"raw file integrity check failed: {checks}")

    # ---- history: one row per event; README: time[i] = gap between behaviour i-1 and i, time[n] = gap to the test query
    hist = (raw.select("user_id",
                       pl.col("history_qry_list").str.split("_").alias("qry"),
                       pl.col("history_wid_list").str.split("_").alias("wid"),
                       pl.col("history_type_list").str.split("_").alias("type"),
                       pl.col("history_time_list").str.split("_").cast(pl.List(pl.Int64)).alias("time"))
            .with_columns(total_time=pl.col("time").list.sum(), time=pl.col("time").list.head(pl.col("wid").list.len()))
            .explode(["qry", "wid", "type", "time"])
            .with_columns(qstr=pl.when(pl.col("qry") == "-1").then(None).otherwise(_canon(pl.col("qry"))))
            .with_columns(qstr=pl.when(pl.col("qstr") == "").then(None).otherwise(pl.col("qstr"))))  # 69 empty queries
    test_q = raw.select("user_id", _canon(pl.col("query")).alias("qstr"))
    queries = (pl.concat([test_q.select("qstr"), hist.select("qstr").drop_nulls()]).unique().sort("qstr")
               .with_row_index("qid")
               .with_columns(qid=pl.col("qid").cast(pl.Int32), terms=pl.col("qstr").str.split(TERM_SEP).cast(pl.List(pl.Int32)))
               .with_columns(n_terms=pl.col("terms").list.len()))
    history = (hist.join(queries.select("qid", "qstr"), on="qstr", how="left")
               .with_columns(step=pl.int_range(pl.len()).over("user_id").cast(pl.Int32),
                             wid=pl.col("wid").cast(pl.Int32),
                             type=pl.col("type").replace_strict(TYPE_CODE, return_dtype=pl.Int8),
                             dt=pl.col("time"),
                             t_before_query=pl.col("total_time") - pl.col("time").cum_sum().over("user_id"))
               .with_columns(age_days=(pl.col("t_before_query") / 86400).cast(pl.Float32))
               .select("user_id", "step", "wid", "qid", "type", "dt", "t_before_query", "age_days")
               .sort("user_id", "step"))
    del hist
    if (history["dt"] < 0).any():
        raise ValueError("negative time gaps in history_time_list")   # ages must be non-increasing within a user
    log.info(f"history: {history.height:,} events, {queries.height:,} distinct queries")

    # ---- candidates: the raw order is sorted by label (positives first) -> shuffle with a fixed seed
    cands = (raw.select("user_id",
                        pl.col("candidate_wid_list").str.split("_").cast(pl.List(pl.Int32)).alias("wid"),
                        pl.col("candidate_label_list").str.split("_").cast(pl.List(pl.Float32)).alias("label"))
             .explode(["wid", "label"])
             .with_columns(label=pl.col("label").cast(pl.Int8), orig_pos=pl.int_range(pl.len()).over("user_id").cast(pl.Int16)))
    rng = np.random.default_rng(seed)
    cands = (cands.with_columns(_key=pl.Series(rng.random(cands.height))).sort("user_id", "_key")
             .with_columns(pos=pl.int_range(pl.len()).over("user_id").cast(pl.Int16))
             .select("user_id", "pos", "wid", "label", "orig_pos"))

    # ---- users, split (by user, among users with a positive) and cross-fitting folds (training users only)
    cs = cands.group_by("user_id").agg(
        n_cand=pl.len().cast(pl.Int16), n_pos=(pl.col("label") > 0).sum().cast(pl.Int16),
        n_click=(pl.col("label") == 1).sum().cast(pl.Int16), n_cart=(pl.col("label") == 2).sum().cast(pl.Int16),
        n_ord=(pl.col("label") == 3).sum().cast(pl.Int16), max_label=pl.col("label").max())
    hs = history.group_by("user_id").agg(hist_len=pl.len().cast(pl.Int32), n_hist_query=pl.col("qid").is_not_null().sum().cast(pl.Int32),
                                         gap_to_query=pl.col("t_before_query").last())
    users = (test_q.join(queries.select("qid", "qstr"), on="qstr", how="left").drop("qstr")
             .join(cs, on="user_id").join(hs, on="user_id").sort("user_id"))
    fr = cfg["prepare"]["split"]
    evaluable = users.filter(pl.col("n_pos") > 0)["user_id"].to_numpy()
    perm = np.random.default_rng(seed).permutation(evaluable)
    n = len(perm); n_tr = int(fr["train"] * n); n_va = int(fr["val"] * n)
    k = int(cfg["prepare"]["n_folds"])
    split_map = pl.DataFrame({"user_id": perm.astype(np.uint32),
                              "split": ["train"] * n_tr + ["val"] * n_va + ["test"] * (n - n_tr - n_va)})
    users = (users.join(split_map, on="user_id", how="left")
             .with_columns(split=pl.col("split").fill_null("train"), has_pos=pl.col("n_pos") > 0))  # no positive -> train
    # fold in 0..k-1 for every training user (out-of-fold tower scores and CTR); -1 for validation / test users
    users = assign_folds(users, k, seed)

    (p.data / "_DONE").unlink(missing_ok=True)
    queries.select("qid", "terms", "n_terms").write_parquet(p.data / "queries.parquet")
    users.write_parquet(p.data / "users.parquet")
    cands.write_parquet(p.data / "candidates.parquet")
    history.write_parquet(p.data / "history.parquet")
    needed = pl.concat([cands.select("wid"), history.select("wid")]).unique()

    # ---- products: 4.8 GB, streamed and restricted to items referenced by the logs
    t1 = time.time()
    (pl.scan_csv(prod_txt, separator="\t", quote_char=None, infer_schema=False)
       .with_columns(pl.col("wid").cast(pl.Int32))
       .join(needed.lazy(), on="wid", how="semi")
       .select("wid", _terms("name").alias("name_terms"), pl.col("brand_id").cast(pl.Int32), _terms("brand_name").alias("brand_terms"),
               *[pl.col(f"cate_id_{i}").cast(pl.Int32) for i in range(1, 5)],
               *[_terms(f"cate_name_{i}").alias(f"cate_terms_{i}") for i in range(1, 5)],
               pl.col("shop_id").cast(pl.Int32))
       .sink_parquet(p.data / "products.parquet"))
    n_prod = pl.scan_parquet(p.data / "products.parquet").select(pl.len()).collect().item()
    log.info(f"products: {n_prod:,} items with metadata in {time.time()-t1:.0f}s")

    front = (cands.filter(pl.col("label") > 0).group_by("user_id")
             .agg(ok=(pl.col("orig_pos").max() == pl.len() - 1)))["ok"].mean()
    stats = {"users": users.height, "users_with_pos": int(users["has_pos"].sum()), "impressions": cands.height,
             "history_events": history.height, "distinct_queries": queries.height, "items_with_metadata": n_prod,
             "split_sizes": dict(users.group_by("split").len().iter_rows()),
             "positives_first_in_raw_order": float(front), "integrity": checks}
    save_json(stats, p.results / "prepare_stats.json")
    (p.data / "_DONE").write_text("ok")
    log.info(f"done in {time.time()-t0:.0f}s: {stats}")


def make_subset(cfg: dict, force: bool = False) -> None:
    """Cut a deterministic user subsample (stratified by split) from the full prepared data."""
    n_users = sample_size(cfg["scope"])
    if n_users is None:
        return
    p = paths(cfg)
    log = get_logger("prepare", p)
    if done(p.data) and not force:
        sync_folds(p.data, cfg, log, src=paths(cfg, "full").data)
        log.info(f"{p.scope} already prepared in {p.data}")
        return
    src = paths(cfg, "full").data
    if not done(src):
        raise FileNotFoundError(f"prepare the full data first ({src})")
    (p.data / "_DONE").unlink(missing_ok=True)
    users = pl.read_parquet(src / "users.parquet")
    rng = np.random.default_rng(cfg["seed"])
    frac = n_users / users.height
    keep = []
    for s in ("train", "val", "test"):
        ids = users.filter(pl.col("split") == s)["user_id"].to_numpy()
        keep.append(rng.choice(ids, size=min(len(ids), math.ceil(len(ids) * frac)), replace=False))
    keep = pl.DataFrame({"user_id": np.sort(np.concatenate(keep)).astype(np.uint32)})
    u = users.join(keep, on="user_id", how="semi")
    c = pl.read_parquet(src / "candidates.parquet").join(keep, on="user_id", how="semi")
    h = pl.read_parquet(src / "history.parquet").join(keep, on="user_id", how="semi")
    w = pl.concat([c.select("wid"), h.select("wid")]).unique()
    q = pl.concat([u.select("qid"), h.select("qid").drop_nulls()]).unique()
    u.write_parquet(p.data / "users.parquet"); c.write_parquet(p.data / "candidates.parquet"); h.write_parquet(p.data / "history.parquet")
    pl.read_parquet(src / "queries.parquet").join(q, on="qid", how="semi").write_parquet(p.data / "queries.parquet")
    pl.scan_parquet(src / "products.parquet").join(w.lazy(), on="wid", how="semi").sink_parquet(p.data / "products.parquet")
    (p.data / "_DONE").write_text("ok")
    log.info(f"{p.scope}: {dict(u.group_by('split').len().iter_rows())} users, {c.height:,} impressions, {h.height:,} events")


def assign_folds(users: pl.DataFrame, k: int, seed: int) -> pl.DataFrame:
    """fold in 0..k-1 for every training user (out-of-fold tower scores and CTR); -1 for validation / test users."""
    is_tr = (users["split"] == "train").to_numpy()
    fold = np.full(users.height, -1, np.int8)
    fold[is_tr] = np.random.default_rng(seed + 1).permutation(int(is_tr.sum())) % max(k, 1)
    return users.with_columns(fold=pl.Series(fold))


def n_folds_of(users: pl.DataFrame) -> int:
    return int(users.filter(pl.col("split") == "train")["fold"].max()) + 1


def sync_folds(data_dir, cfg: dict, log, src=None) -> None:
    """Re-assign folds in place when `prepare.n_folds` changed since the data was prepared (cheap: users table only),
    giving the same assignment a fresh prepare would. A subsample (`src` = the full data dir) copies the full folds."""
    users = pl.read_parquet(data_dir / "users.parquet")
    k = int(cfg["prepare"]["n_folds"])
    if n_folds_of(users) == k:
        return
    log.info(f"{data_dir}: re-assigning {n_folds_of(users)} -> {k} cross-fitting folds "
             "(downstream stages then refuse their old outputs: re-run them with --force)")
    if src is None:
        users = assign_folds(users.sort("user_id"), k, cfg["seed"])
    else:
        full = pl.read_parquet(src / "users.parquet", columns=["user_id", "fold"])
        users = users.drop("fold").join(full, on="user_id", how="left").select(users.columns)
    users.write_parquet(data_dir / "users.parquet")


def check_folds(users: pl.DataFrame, cfg: dict) -> None:
    """Fail loudly when the prepared folds no longer match `prepare.n_folds` (e.g. data prepared with 2 folds, config now
    5): cross-fitting would otherwise silently keep the old fold count and leave the extra folds empty."""
    k = int(cfg["prepare"]["n_folds"])
    have = n_folds_of(users)
    if have != k:
        raise ValueError(f"prepared data has {have} folds but prepare.n_folds = {k}: re-run "
                         f"`00_prepare.py` (re-assigns the folds), then the later stages with --force")


def load(data_dir, name: str, columns=None) -> pl.DataFrame:
    return pl.read_parquet(data_dir / f"{name}.parquet", columns=columns)
