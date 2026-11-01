"""Stage 5: LightGBM LambdaRank re-ranker over hand-crafted features + cross-fitted tower scores.

Trained on training users' slates (graded labels, gain 2^label - 1), early-stopped on validation users.
Tower scores of training users are out-of-fold, so the ranker learns how much to trust a tower score that was not
fitted on the same slate -- the situation it meets on validation and test users.
"""
from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np
import polars as pl

from ..config import check_fresh, get_logger, paths, save_json
from . import dataset


def fit(cfg: dict, p, names: list[str], use_tower: bool = True, user_frac: float | None = None, tag: str = "ranker",
        log=None) -> lgb.Booster:
    """Load the rows and fit one ranker, on a share of training users when `user_frac` is set (the ablations do this).

    Slates without a positive are dropped: they have no pair to order, so they give LambdaRank nothing to learn from
    and leave NDCG undefined on validation.
    """
    rc = cfg["ranker"]
    log = log or get_logger("ranker", p)
    frac = rc["train_user_frac"] if user_frac is None else user_frac
    t0 = time.time()
    tr = dataset.load(p, ["train"], cfg, use_tower, user_frac=frac, only_with_pos=True)
    va = dataset.load(p, ["val"], cfg, use_tower, only_with_pos=True)
    log.info(f"[{tag}] loaded in {time.time()-t0:.0f}s")
    return fit_frames(cfg, tr, va, names, tag=tag, log=log)


def fit_frames(cfg: dict, tr: pl.DataFrame, va: pl.DataFrame, names: list[str], tag: str = "ranker", log=None) -> lgb.Booster:
    """LambdaRank on already-loaded training / validation rows (sorted by user, positives-only slates).

    Each user's slate is one query group. Boosting stops once validation NDCG has not improved for
    `early_stopping_rounds`, which sets the number of trees.
    """
    rc = cfg["ranker"]
    log = log or get_logger("ranker")
    t0 = time.time()
    log.info(f"[{tag}] train {tr['user_id'].n_unique():,} slates / {tr.height:,} rows, val {va['user_id'].n_unique():,} slates, "
             f"{len(names)} features")
    params = {**rc["params"], "label_gain": rc["label_gain"], "seed": cfg["seed"], "deterministic": True,
              "force_row_wise": True}
    dtr = lgb.Dataset(tr.select(names).to_numpy(), tr["label"].to_numpy(), group=dataset.groups(tr), feature_name=names,
                      free_raw_data=True)
    dva = lgb.Dataset(va.select(names).to_numpy(), va["label"].to_numpy(), group=dataset.groups(va), reference=dtr)
    del tr, va
    evals = {}
    booster = lgb.train(params, dtr, num_boost_round=rc["num_boost_round"], valid_sets=[dva], valid_names=["val"],
                        callbacks=[lgb.early_stopping(rc["early_stopping_rounds"], verbose=False),
                                   lgb.log_evaluation(max(1, rc["num_boost_round"] // 20)), lgb.record_evaluation(evals)])
    log.info(f"[{tag}] best iteration {booster.best_iteration}, val {dict(booster.best_score['val'])} ({time.time()-t0:.0f}s)")
    return booster


def train(cfg: dict, force: bool = False) -> None:
    """Fit the final ranker and save it at its best iteration, with the feature importances and feature order."""
    p = paths(cfg)
    log = get_logger("ranker", p)
    out = p.ranker / "ranker.txt"
    if out.exists() and not force:
        check_fresh(out, p.features / "feature_list.json", p.tower / "tower_scores.parquet")
        log.info("ranker exists (use --force to retrain)")
        return
    names = dataset.feature_names(p)
    booster = fit(cfg, p, names, log=log)
    booster.save_model(str(out), num_iteration=booster.best_iteration)
    imp = pl.DataFrame({"feature": names, "gain": booster.feature_importance("gain"),
                        "split": booster.feature_importance("split").astype(np.int64)}).sort("gain", descending=True)
    imp.write_csv(p.results / "ranker_feature_importance.csv")
    save_json({"best_iteration": booster.best_iteration, "features": names}, p.ranker / "ranker_meta.json")
    log.info("top features by gain:\n" + str(imp.head(15)))
