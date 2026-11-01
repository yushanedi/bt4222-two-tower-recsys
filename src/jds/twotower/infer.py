"""Two-tower inference: cross-fitted scores for every displayed item -> <scope>/tower/tower_scores.parquet.

Training users are scored by the fold model that did not train on them (out-of-fold); validation and test users by
the `full` model. The ranker consumes these scores as features.
"""
from __future__ import annotations

import polars as pl

from ..config import check_fresh, get_device, get_logger, paths, save_json
from ..metrics import evaluate, results_table
from ..prepare import load
from .train import ItemFeats, load_assets, load_model, make_stores, model_plan, score_users


def score(cfg: dict, variant: str = "", force: bool = False) -> pl.DataFrame:
    """Score every displayed item with the model `model_plan` assigns to its user, save the scores and tower metrics.

    An ablation `variant` only has a `full` model, so only validation and test users are scored. Existing scores are
    reused unless `force`, after checking that they are newer than the data and checkpoints they came from.
    """
    p = paths(cfg)
    log = get_logger("tower", p)
    out = p.tower / (f"tower_scores_{variant}.parquet" if variant else "tower_scores.parquet")
    if out.exists() and not force:
        names = [f"full_{variant}"] if variant else [n for n, _, _ in model_plan(cfg, load(p.data, "users"))]
        cks = [p.tower / f"{n}.pt" for n in names]
        check_fresh(out, p.data / "users.parquet", *cks)
        log.info(f"{out.name} exists (use --force to rescore)")
        return pl.read_parquet(out)
    device = get_device(cfg)
    a = load_assets(p)
    st, users = make_stores(cfg, p, a)
    feats = ItemFeats(a["X"], device, cfg["tower"]["item_matrix_on_gpu"] and device == "cuda")
    parts = []
    for name, _, score_users_ in model_plan(cfg, users):
        if variant:
            if name != "full":
                continue
            name = f"full_{variant}"
            score_users_ = users.filter(pl.col("split").is_in(["val", "test"]))["user_id"].to_numpy()
        ck = p.tower / f"{name}.pt"
        if not ck.exists():
            raise FileNotFoundError(f"{ck} missing: run scripts/03_train_tower.py first")
        model = load_model(ck, a, device)
        s = score_users(model, st, feats, score_users_, cfg, device).with_columns(tower_model=pl.lit(name))
        log.info(f"[{name}] scored {s['user_id'].n_unique():,} users / {s.height:,} rows")
        parts.append(s)
    df = pl.concat(parts).join(users.select("user_id", "split"), on="user_id")
    df.write_parquet(out)

    rows = []
    for sp in ("train", "val", "test"):
        d = df.filter(pl.col("split") == sp)
        if d.height and d["label"].max() > 0:
            label = "train (out-of-fold)" if sp == "train" else sp
            rows.append(evaluate(d, "tower_score", seed=cfg["seed"], name=f"two-tower{'-' + variant if variant else ''} [{label}]"))
    save_json(rows, p.results / (f"tower_{variant}.json" if variant else "tower.json"))
    log.info("\n" + str(results_table(rows)))
    return df
