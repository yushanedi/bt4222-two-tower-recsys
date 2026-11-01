"""Stage 7: where the model helps and why -> <scope>/results/diag_*.csv|json and figures/diag_*.png

  * segments  - test metrics by user / query segment (activity in the click window, history length, query frequency,
                query length, whether the positives are cold items, deepest label), incl. the prior team's model
  * insights  - lift of behavioural signals (open carts, repurchase, recent clicks, ...) on engagement
  * ablations - rankers retrained without one feature group at a time (same user subsample for all)
  * shap      - LightGBM's exact TreeSHAP contributions (pred_contrib) on test rows, aggregated by feature and group
  * attention - how the full tower splits candidate-aware attention between short-window events and the null slot
"""
from __future__ import annotations

import sys

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402

from . import features  # noqa: E402
from .config import get_device, get_logger, load_json, paths, save_json  # noqa: E402
from .metrics import evaluate, per_user_metrics, results_table  # noqa: E402
from .prepare import load  # noqa: E402
from .ranker import dataset  # noqa: E402
from .ranker import infer as rinfer  # noqa: E402
from .ranker import train as rtrain  # noqa: E402

GROUPS = {   # feature-name prefixes per group (the distinct data-field roles in features.py)
    "two-tower score": ("tower_",),
    "personal history (per type window)": ("u_wid_", "u_brand_id_", "u_shop_id_", "u_cate_id_4_", "u_cart_not_ordered"),
    "similar past queries": ("u_q_", "u_simq_"),
    "query -> item history (all users)": ("qpop_",),
    "impression CTR (out-of-fold)": ("ctr_", "gain_", "disp_"),
    "popularity (per type window)": ("ipop_", "apop_"),
    "lexical": ("lex_",),
    "slate context": ("slate_",),
    "user activity": ("ua_",),
    "item content": ("item_",),
}


def group_of(name: str) -> str:
    base = name[:-4] if name.endswith("_pct") and not name.startswith("tower_") else name
    for g, pre in GROUPS.items():
        if base.startswith(pre):
            return g
    return "other"


def segment_table(cfg, p) -> pl.DataFrame:
    """One row per evaluable test user with the segment it belongs to in each dimension."""
    users = load(p.data, "users")
    test = users.filter((pl.col("split") == "test") & pl.col("has_pos"))
    tr_q = users.filter(pl.col("split") == "train").group_by("qid").len().rename({"len": "q_train_users"})
    hist = load(p.data, "history", ["user_id", "wid", "type", "age_days"])
    act = hist.group_by("user_id").agg(clicks_7=((pl.col("type") == 1) & (pl.col("age_days") < 7)).sum())
    seen = hist.select("wid").unique().with_columns(seen=pl.lit(True))
    pos = (load(p.data, "candidates", ["user_id", "wid", "label"]).filter(pl.col("label") > 0)
           .join(test.select("user_id"), on="user_id", how="semi").join(seen, on="wid", how="left")
           .group_by("user_id").agg(any_seen=pl.col("seen").fill_null(False).any(), max_label=pl.col("label").max()))
    qlen = load(p.data, "queries", ["qid", "n_terms"])
    return (test.select("user_id", "qid", "hist_len").join(tr_q, on="qid", how="left").join(act, on="user_id", how="left")
            .join(pos, on="user_id", how="left").join(qlen, on="qid", how="left")
            .with_columns(pl.col("q_train_users", "clicks_7", "n_terms").fill_null(0))
            .select("user_id",
                    activity=pl.when(pl.col("clicks_7") > 0).then(pl.lit("clicked in last 7 days")).otherwise(pl.lit("no click in last 7 days")),
                    history=pl.when(pl.col("hist_len") < 50).then(pl.lit("history < 50")).when(pl.col("hist_len") < 200)
                    .then(pl.lit("history 50-199")).otherwise(pl.lit("history >= 200")),
                    query=pl.when(pl.col("q_train_users") == 0).then(pl.lit("query unseen in training"))
                    .when(pl.col("q_train_users") < 10).then(pl.lit("query seen 1-9x")).otherwise(pl.lit("query seen >= 10x")),
                    query_length=pl.when(pl.col("n_terms") <= 2).then(pl.lit("1-2 terms")).when(pl.col("n_terms") <= 4)
                    .then(pl.lit("3-4 terms")).otherwise(pl.lit("5+ terms")),
                    positive_items=pl.when(pl.col("any_seen")).then(pl.lit("some positive seen in history"))
                    .otherwise(pl.lit("all positives cold")),
                    deepest_label=pl.col("max_label").replace_strict({1: "best = click", 2: "best = cart", 3: "best = order"},
                                                                      default="?", return_dtype=pl.String)))


SEG_DIMS = ("activity", "history", "query", "query_length", "positive_items", "deepest_label")


def segments(cfg, p, log) -> pl.DataFrame:
    seg = segment_table(cfg, p)
    models = {"ranker": pl.read_parquet(p.results / "per_user_ranker_test.parquet")}
    ts = pl.read_parquet(p.tower / "tower_scores.parquet").filter(pl.col("split") == "test")
    models["two-tower"] = per_user_metrics(ts, "tower_score", seed=cfg["seed"])
    pop = features.read(p, columns=["user_id", "label", "ipop_users"], splits=["test"])
    models["popularity"] = per_user_metrics(pop, "ipop_users", seed=cfg["seed"])
    f = p.results / "per_user_prior_ncf_fixed_test.parquet"
    if f.exists():
        models["prior NCF (fixed)"] = pl.read_parquet(f)
    rows = []
    for dim in SEG_DIMS:
        for name, pu in models.items():
            d = pu.join(seg, on="user_id").group_by(dim).agg(n=pl.len(), mrr=pl.col("mrr").mean(), gndcg10=pl.col("gndcg10").mean())
            rows.append(d.rename({dim: "segment"}).with_columns(dimension=pl.lit(dim), model=pl.lit(name)))
    out = pl.concat(rows).select("dimension", "segment", "model", "n", "mrr", "gndcg10").sort("dimension", "segment", "model")
    out.write_csv(p.results / "diag_segments.csv")
    _plot_segments(out, list(models), p.figures / "diag_segments.png")
    log.info("segments:\n" + str(out))
    return out


def _plot_segments(out: pl.DataFrame, models: list[str], path) -> None:
    dims = [d for d in SEG_DIMS if d in out["dimension"].unique().to_list()]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#8172B2"]
    for ax, dim in zip(axes.flat, dims):
        d = out.filter(pl.col("dimension") == dim)
        segs = sorted(d["segment"].unique().to_list())
        x = np.arange(len(segs))
        w = 0.8 / len(models)
        for j, m in enumerate(models):
            v = [d.filter((pl.col("segment") == s) & (pl.col("model") == m))["gndcg10"].mean() or 0 for s in segs]
            ax.bar(x + j * w, v, w, color=colors[j % len(colors)], label=m)
        n = [int(d.filter(pl.col("segment") == s)["n"].max()) for s in segs]
        ax.set_xticks(x + w * (len(models) - 1) / 2, [f"{s}\n(n={k:,})" for s, k in zip(segs, n)], fontsize=7)
        ax.set(title=dim.replace("_", " "), ylabel="graded NDCG@10")
    axes.flat[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


INSIGHT_SIGNALS = {   # behavioural signal -> boolean expression on the feature table
    "item carted in last 90 d, not ordered since": pl.col("u_cart_not_ordered") > 0,
    "item ordered by the user in last 365 d (repurchase)": pl.col("u_wid_order_365") > 0,
    "item clicked by the user in last 7 d": pl.col("u_wid_click_7") > 0,
    "brand ordered by the user in last 365 d": pl.col("u_brand_id_order_365") > 0,
    "shop ordered by the user in last 365 d": pl.col("u_shop_id_order_365") > 0,
    "category clicked by the user in last 7 d": pl.col("u_cate_id_4_click_7") > 0,
    "item engaged under a similar past query": pl.col("u_simq_wid_n") > 0,
    "item never in any user's history (cold)": pl.col("ipop_users") == 0,
}


def insights(cfg, p, log) -> pl.DataFrame:
    """Lift of behavioural signals on engagement in the test session (all displayed items of all users).

    For each signal: how many displayed items carry it, the positive / order rate with and without it, and the share
    of all positives (and of all orders) it covers. Labels are only described here, never used as features.
    """
    cols = ["user_id", "label", "u_cart_not_ordered", "u_wid_order_365", "u_wid_click_7", "u_brand_id_order_365",
            "u_shop_id_order_365", "u_cate_id_4_click_7", "u_simq_wid_n", "ipop_users"]
    df = features.read(p, columns=cols)
    pos, ordr = pl.col("label") > 0, pl.col("label") == 3
    n_pos, n_ord = df.select(pos.sum()).item(), max(df.select(ordr.sum()).item(), 1)
    rows = []
    for name, cond in INSIGHT_SIGNALS.items():
        r = df.select(share_of_displayed=cond.mean(),
                      pos_rate_with=pos.filter(cond).mean(), pos_rate_without=pos.filter(~cond).mean(),
                      order_rate_with=ordr.filter(cond).mean(), order_rate_without=ordr.filter(~cond).mean(),
                      share_of_positives=(cond & pos).sum() / max(n_pos, 1), share_of_orders=(cond & ordr).sum() / n_ord).row(0, named=True)
        r = {k: (float(v) if v is not None else None) for k, v in r.items()}
        r["pos_lift"] = r["pos_rate_with"] / r["pos_rate_without"] if r["pos_rate_with"] is not None and r["pos_rate_without"] else None
        rows.append({"signal": name, **r})
    out = pl.DataFrame(rows).with_columns(pl.col(pl.Float64).round(4))
    out.write_csv(p.results / "diag_insights.csv")
    log.info("behavioural signals vs engagement in the test session:\n" + str(out))
    return out


def ablations(cfg, p, log) -> pl.DataFrame:
    """Retrain the ranker without each feature group in turn and compare test metrics with the all-features ranker.

    All rankers use the same user subsample, so the differences reflect the feature group, not the training data.
    """
    frac = cfg["diagnostics"]["ablation_user_frac"]
    names = dataset.feature_names(p)
    rows = []
    for g in ["(none)", *GROUPS]:
        keep = names if g == "(none)" else [n for n in names if group_of(n) != g]
        b = rtrain.fit(cfg, p, keep, user_frac=frac, tag=f"ablation: {g}", log=log)
        pred = rinfer.predict(cfg, p, b, keep, "test")
        rows.append(evaluate(pred, "score", seed=cfg["seed"], name=f"without {g}" if g != "(none)" else "all features"))
    tbl = results_table(rows)
    tbl.write_csv(p.results / "diag_ablations.csv")
    log.info(f"ablations (rankers on {frac:.0%} of training users):\n{tbl}")
    return tbl


def shap(cfg, p, log) -> pl.DataFrame:
    """Mean |SHAP| of the final ranker on a sample of test rows, per feature and summed per feature group."""
    booster = lgb.Booster(model_file=str(p.ranker / "ranker.txt"))
    names = load_json(p.ranker / "ranker_meta.json")["features"]
    df = dataset.load(p, ["test"], cfg)
    n = min(cfg["diagnostics"]["shap_rows"], df.height)
    X = df.sample(n, seed=cfg["seed"]).select(names).to_numpy()
    contrib = booster.predict(X, pred_contrib=True)[:, :-1]          # last column is the expected value
    imp = (pl.DataFrame({"feature": names, "mean_abs_shap": np.abs(contrib).mean(0)})
           .with_columns(group=pl.col("feature").map_elements(group_of, return_dtype=pl.String))
           .sort("mean_abs_shap", descending=True))
    imp.write_csv(p.results / "diag_shap_features.csv")
    grp = imp.group_by("group").agg(pl.col("mean_abs_shap").sum()).sort("mean_abs_shap", descending=True)
    grp.write_csv(p.results / "diag_shap_groups.csv")
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    top = imp.head(20).reverse()
    ax[0].barh(top["feature"].to_list(), top["mean_abs_shap"].to_numpy(), color="#4C72B0")
    ax[0].set(title="Top 20 features (mean |SHAP|, test rows)")
    g = grp.reverse()
    ax[1].barh(g["group"].to_list(), g["mean_abs_shap"].to_numpy(), color="#DD8452")
    ax[1].set(title="Feature groups (sum of mean |SHAP|)")
    fig.tight_layout()
    fig.savefig(p.figures / "diag_shap.png", dpi=150)
    plt.close(fig)
    log.info("SHAP by group:\n" + str(grp))
    return grp


@torch.no_grad()
def attention(cfg, p, log) -> dict:
    from .twotower.data import SEGMENTS
    from .twotower.train import ItemFeats, load_assets, load_model, make_stores, to_dev
    device = get_device(cfg)
    a = load_assets(p)
    st, users = make_stores(cfg, p, a)
    model = load_model(p.tower / "full.pt", a, device)
    feats = ItemFeats(a["X"], device, cfg["tower"]["item_matrix_on_gpu"] and device == "cuda")
    ids = users.filter(pl.col("split") == "test")["user_id"].to_numpy()[: cfg["diagnostics"]["attention_users"]]
    # Only the short window is attended in the current architecture; mid/long are summarised by per-type counts.
    # Attention is candidate-aware: average the null-slot mass over the real candidates of each slate.
    null_mass, nonempty = [], []
    for i in range(0, len(ids), cfg["tower"]["eval_batch_users"]):
        ub = ids[i:i + cfg["tower"]["eval_batch_users"]]
        b, items, _ = st.slate_batch(ub)
        b = to_dev(b, device)
        it = torch.as_tensor(items, device=device)
        rows = feats(it)
        model.user_vec(b["q"], b["ctx"], feats, it, cand_rows=rows, cand_v=model.item(rows))
        att = model.user.attn.last_attn.float().cpu().numpy()          # [B,S,L+1]
        keep = (items > 0)                                             # real candidate slots
        null_mass.append((att[..., 0] * keep).sum(1) / keep.sum(1).clip(1))
        nonempty.append(b["ctx"]["short"]["mask"].any(-1).cpu().numpy())
    nm, ne = np.concatenate(null_mass), np.concatenate(nonempty)
    out = {"short": {"share_users_with_events": float(ne.mean()),
                     "mean_attention_on_events_when_present": float(1 - nm[ne].mean()) if ne.any() else None},
           "mid": {"attention": "none (per-type counts only)"},
           "long": {"attention": "none (per-type counts only)"}}
    save_json(out, p.results / "diag_attention.json")
    log.info(f"tower attention by history segment: {out}")
    return out


def run(cfg: dict, parts: list[str]) -> None:
    p = paths(cfg)
    log = get_logger("diagnostics", p)
    if "attention" in parts:          # PyTorch before LightGBM: two OpenMP runtimes in one process can deadlock on macOS
        if sys.platform == "darwin":
            torch.set_num_threads(1)
        attention(cfg, p, log)
    if "segments" in parts:
        segments(cfg, p, log)
    if "shap" in parts:
        shap(cfg, p, log)
    if "insights" in parts:
        insights(cfg, p, log)
    if "ablations" in parts:
        ablations(cfg, p, log)
    for v in ("no_history", "no_query"):
        f = p.results / f"tower_{v}.json"
        if f.exists():
            log.info(f"tower ablation {v}: " + str([(r["model"], round(r["mrr"], 4)) for r in load_json(f)]))
    log.info("\n" + str(rinfer.summary(cfg)))
