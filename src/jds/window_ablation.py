"""Stage 2d: does counting history inside each type's retention window actually help? (interim feedback)

Two ways of turning the same history into features:
  * windowed (the pipeline): every count is per event type inside that type's own retention window
    (click_7, cart_90, order_365, ...), plus days since the last event;
  * naive (what you would build without noticing the windows): counts per type over the whole history, a
    time-decayed score that mixes types (click 1, cart 2, order 3), and per-user click / cart / order ratios
    (the previous team's user statistics).

Everything else (query -> item, CTR, lexical, slate context, item content) is identical, and no tower score is
used, so the comparison runs on CPU straight after stage 2. Rankers are trained on the same user subsample:

  windowed           the pipeline's features
  naive              windowed history columns replaced by the naive ones
  windowed + ratios  the pipeline's features + the naive user ratios (do the ratios add anything?)
  no history counts  neither (lower bound: how much do history counts matter at all?)

Outputs: <scope>/features_naive/part_*.parquet, results/window_ablation.csv, results/window_ablation_segments.csv,
figures/window_ablation.png
"""
from __future__ import annotations

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from . import features  # noqa: E402
from .config import get_logger, load_json, paths, save_json  # noqa: E402
from .metrics import evaluate, per_user_metrics, results_table  # noqa: E402
from .prepare import load  # noqa: E402
from .ranker import train as rtrain  # noqa: E402
from .windows import CART, CLICK, FLW, ORD, feature_windows  # noqa: E402

INF = float("inf")
NAIVE_WINS = [("click_all", CLICK, INF), ("cart_all", CART, INF), ("order_all", ORD, INF), ("follow_all", FLW, INF)]
TYPE_WEIGHT = {CLICK: 1.0, FLW: 1.0, CART: 2.0, ORD: 3.0}
HALF_LIFE_DAYS = 30.0


def windowed_history_columns(names: list[str], cfg: dict) -> list[str]:
    """Feature columns that exist only because history is counted per type window (what the naive variant replaces)."""
    suffixes = tuple(f"_{n}" for n, _, _ in feature_windows(cfg))
    hist_prefix = ("u_", "ipop_", "apop_", "qpop_", "ua_")
    out = []
    for n in names:
        base = n[:-4] if n.endswith("_pct") else n
        if base.startswith(("u_q_", "u_simq_")):          # similar-query features are not per-type counts: kept in both
            continue
        if base.startswith(hist_prefix) and (base.endswith(suffixes) or base.endswith("_last_days") or base == "u_cart_not_ordered"):
            out.append(n)
    return out


def _decay() -> pl.Expr:
    """Sum of type weights (click 1, cart 2, order 3) decayed with a 30-day half-life: mixes types across windows."""
    w = pl.col("type").replace_strict(TYPE_WEIGHT, return_dtype=pl.Float32)
    return (w * (-np.log(2) * pl.col("age_days") / HALF_LIFE_DAYS).exp()).sum().cast(pl.Float32)


def _naive_counts(df: pl.DataFrame, by: list[str], prefix: str) -> pl.DataFrame:
    return df.group_by(by).agg([(pl.col("type") == t).sum().cast(pl.Float32).alias(f"{prefix}_{n}") for n, t, _ in NAIVE_WINS]
                               + [_decay().alias(f"{prefix}_decay")])


def naive_tables(history: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Global popularity tables with no windows (all history, per type, plus a decayed mixed-type score)."""
    out = {"wid": _naive_counts(history, ["wid"], "nv_ipop")}
    for k in features.ATTR_KEYS:
        out[k] = _naive_counts(history.drop_nulls(k), [k], f"nv_apop_{k}")
    hq = history.filter(pl.col("qid").is_not_null())
    for k in ("wid", "brand_id", "cate_id_4"):
        out[f"q_{k}"] = _naive_counts(hq.drop_nulls(k), ["qid", k], f"nv_qpop_{k}")
    return out


def naive_user(hist: pl.DataFrame) -> pl.DataFrame:
    """Previous-team style user statistics: type ratios over the whole history."""
    return hist.group_by("user_id").agg(
        nv_ua_click_ratio=(pl.col("type") == CLICK).mean().cast(pl.Float32),
        nv_ua_cart_ratio=(pl.col("type") == CART).mean().cast(pl.Float32),
        nv_ua_order_ratio=(pl.col("type") == ORD).mean().cast(pl.Float32),
        **{f"nv_ua_{n}": (pl.col("type") == t).sum().cast(pl.Float32) for n, t, _ in NAIVE_WINS})


def build(cfg: dict, force: bool = False) -> list[str]:
    """Naive history features for every displayed (user, item), aligned with the main feature parts."""
    p = paths(cfg)
    out_dir = p.root / "features_naive"
    out_dir.mkdir(exist_ok=True)
    log = get_logger("window_ablation", p)
    if (out_dir / "feature_list.json").exists() and not force:
        return load_json(out_dir / "feature_list.json")
    for old in out_dir.glob("part_*.parquet"):
        old.unlink()
    t0 = time.time()
    users = load(p.data, "users", ["user_id", "qid"])
    cands = load(p.data, "candidates", ["user_id", "wid"])
    attrs = pl.scan_parquet(p.data / "products.parquet").select("wid", *features.ATTR_KEYS).collect()
    history = load(p.data, "history", ["user_id", "wid", "qid", "type", "age_days"]).join(attrs, on="wid", how="left")
    glob = naive_tables(history)
    ids = users["user_id"].to_numpy()
    step = cfg["features"]["chunk_users"]
    names = None
    for i, s in enumerate(range(0, len(ids), step)):
        chunk = pl.DataFrame({"user_id": ids[s:s + step]})
        r = cands.join(chunk, on="user_id", how="semi").join(users, on="user_id").join(attrs, on="wid", how="left")
        h = history.join(chunk, on="user_id", how="semi")
        r = r.join(glob["wid"], on="wid", how="left")
        for k in features.ATTR_KEYS:
            r = r.join(glob[k], on=k, how="left")
        for k in ("wid", "brand_id", "cate_id_4"):
            r = r.join(glob[f"q_{k}"], on=["qid", k], how="left")
        for k in ["wid", *features.ATTR_KEYS]:
            r = r.join(_naive_counts(h.drop_nulls(k), ["user_id", k], f"nv_u_{k}"), on=["user_id", k], how="left")
        r = r.join(naive_user(h), on="user_id", how="left")
        cols = [c for c in r.columns if c.startswith("nv_")]
        r = r.with_columns([pl.col(c).fill_null(0.0) for c in cols])
        n = pl.len().over("user_id")
        pct = ["nv_ipop_click_all", "nv_ipop_decay", "nv_qpop_wid_click_all", "nv_u_cate_id_4_click_all"]
        r = r.with_columns([((pl.col(c).rank("average").over("user_id") - 1) / (n - 1).clip(1)).cast(pl.Float32).alias(f"{c}_pct")
                            for c in pct])
        names = names or [c for c in r.columns if c.startswith("nv_")]
        r.select("user_id", "wid", *names).sort("user_id", "wid").write_parquet(out_dir / f"part_{i:03d}.parquet")
    save_json(names, out_dir / "feature_list.json")
    log.info(f"naive features: {len(names)} columns ({time.time()-t0:.0f}s)")
    return names


def _load(p, splits, cfg, frac, only_with_pos=True) -> pl.DataFrame:
    df = features.read(p, splits=splits, user_frac=frac, seed=cfg["seed"])
    if only_with_pos:
        df = df.filter((pl.col("label") > 0).any().over("user_id"))
    nv = pl.scan_parquet(p.root / "features_naive" / "part_*.parquet").join(df.select("user_id", "wid").lazy(), on=["user_id", "wid"],
                                                                           how="semi").collect()
    return df.join(nv, on=["user_id", "wid"], how="left").sort("user_id", "wid")


def variants(names: list[str], naive: list[str], cfg: dict) -> dict[str, list[str]]:
    """Feature list of each compared ranker (the four variants in the module docstring)."""
    win = set(windowed_history_columns(names, cfg))
    ratios = [n for n in naive if n.endswith("_ratio")]
    base = [n for n in names if n not in win]
    return {"windowed (pipeline)": names, "naive (all-history counts, decay, ratios)": base + naive,
            "windowed + naive ratios": names + ratios, "no history counts": base}


def history_age_segment(p) -> pl.DataFrame:
    """Test users by how much of their history is older than the click window (where naive counts mislead most)."""
    h = load(p.data, "history", ["user_id", "age_days"])
    return (h.group_by("user_id").agg(old=(pl.col("age_days") >= 7).mean())
            .select("user_id", segment=pl.when(pl.col("old") < 0.5).then(pl.lit("< 50% of history older than 7 d"))
                    .when(pl.col("old") < 0.9).then(pl.lit("50-90% older than 7 d")).otherwise(pl.lit(">= 90% older than 7 d"))))


def run(cfg: dict, force: bool = False) -> pl.DataFrame:
    """Train one ranker per variant on the same users, then compare test metrics overall and by history age."""
    p = paths(cfg)
    log = get_logger("window_ablation", p)
    naive = build(cfg, force)
    names = load_json(p.features / "feature_list.json")
    frac = cfg["diagnostics"]["ablation_user_frac"]
    tr = _load(p, ["train"], cfg, frac)
    va = _load(p, ["val"], cfg, 1.0)
    te = _load(p, ["test"], cfg, 1.0, only_with_pos=False)
    seg = history_age_segment(p)
    rows, seg_rows = [], []
    for name, cols in variants(names, naive, cfg).items():
        b = rtrain.fit_frames(cfg, tr, va, cols, tag=f"window ablation: {name}", log=log)
        pred = te.select("user_id", "wid", "label").with_columns(score=pl.Series(b.predict(te.select(cols).to_numpy())).cast(pl.Float32))
        rows.append({**evaluate(pred, "score", seed=cfg["seed"], name=name), "n_features": len(cols)})
        pu = per_user_metrics(pred, "score", seed=cfg["seed"]).join(seg, on="user_id", how="left")
        seg_rows.append(pu.group_by("segment").agg(n=pl.len(), mrr=pl.col("mrr").mean(), gndcg10=pl.col("gndcg10").mean())
                        .with_columns(model=pl.lit(name)))
    tbl = results_table(rows).join(pl.DataFrame(rows).select("model", "n_features"), on="model")
    tbl.write_csv(p.results / "window_ablation.csv")
    segs = pl.concat(seg_rows).select("segment", "model", "n", "mrr", "gndcg10").sort("segment", "model")
    segs.write_csv(p.results / "window_ablation_segments.csv")
    _plot(tbl, segs, p.figures / "window_ablation.png", frac)
    log.info(f"window ablation (rankers on {frac:.0%} of training users, no tower score):\n{tbl}\n{segs}")
    return tbl


SEG_ORDER = ["< 50% of history older than 7 d", "50-90% older than 7 d", ">= 90% older than 7 d"]
SEG_SHORT = {"< 50% of history older than 7 d": "< 50% older\nthan 7 d", "50-90% older than 7 d": "50-90% older\nthan 7 d",
             ">= 90% older than 7 d": ">= 90% older\nthan 7 d"}


def _plot(tbl: pl.DataFrame, segs: pl.DataFrame, path, frac) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"width_ratios": [1, 1.1]})
    models = tbl["model"].to_list()
    colors = ["#4C72B0", "#DD8452", "#55A868", "#8C8C8C"]
    ax[0].barh(models[::-1], tbl["gndcg10"].to_numpy()[::-1], color=colors[: len(models)][::-1])
    for i, v in enumerate(tbl["gndcg10"].to_numpy()[::-1]):
        ax[0].text(v, i, f" {v:.4f}", va="center", fontsize=8)
    ax[0].set(xlabel="graded NDCG@10 (test users)", title=f"Same ranker, different history features\n(trained on {frac:.0%} of training users)")
    present = set(segs["segment"].unique().to_list())
    seg_names = [s_ for s_ in SEG_ORDER if s_ in present]
    x = np.arange(len(seg_names))
    w = 0.8 / len(models)
    for j, m in enumerate(models):
        d = segs.filter(pl.col("model") == m)
        v = [d.filter(pl.col("segment") == s_)["gndcg10"].mean() or 0 for s_ in seg_names]
        ax[1].bar(x + j * w, v, w, color=colors[j % len(colors)], label=m)
    n = [int(segs.filter(pl.col("segment") == s_)["n"].max()) for s_ in seg_names]
    ax[1].set_xticks(x + w * (len(models) - 1) / 2, [f"{SEG_SHORT[s_]}\n(n={k:,})" for s_, k in zip(seg_names, n)], fontsize=8)
    ax[1].set(ylabel="graded NDCG@10 (test users)", title="By share of the user's history older than the 7-day click window")
    ax[1].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
