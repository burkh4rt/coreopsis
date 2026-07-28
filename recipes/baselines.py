#!/usr/bin/env python3

"""
baseline model
"""

import fnmatch
import importlib.resources as resources
import os
import pathlib

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from omegaconf import OmegaConf
from sklearn import metrics as skl_mets

pd.options.display.float_format = "{:,.3f}".format
pd.options.display.max_columns = None
pd.options.display.max_rows = 100
pd.options.display.width = None
pd.options.display.expand_frame_repr = False
pd.options.display.show_dimensions = True
pd.set_option("performance_warnings", False)

hm = (
    pathlib.Path("/gpfs/data" if os.uname().nodename.startswith("cri") else "/mnt")
    / "bbj-lab/users/burkh4rt"
)

dsets = ("ucmc-icu", "nu-icu", "mimic-icu")

grokked_outcome_tokens = [
    x
    for x in OmegaConf.load(
        hm / "processed" / dsets[0] / "tokenizer.yaml"
    ).lookup.keys()
    if any(
        fnmatch.fnmatch(x, p)
        for p in OmegaConf.load(resources.files("coreopsis.config") / "scoring.yaml")[
            "tokens_of_interest"
        ]
    )
]

pt_cols = OmegaConf.load(resources.files("coreopsis.config") / "collation.yaml")[
    "pass_through_columns"
]

vocab = OmegaConf.load(hm / "processed" / dsets[0] / "tokenizer.yaml").lookup


bl_roc_auc = pd.DataFrame(index=grokked_outcome_tokens, columns=dsets)
bl_pr_auc = bl_roc_auc.copy()

for ds in dsets:
    for tt in grokked_outcome_tokens:
        lf_train, lf_tuning, lf_held_out = (
            pl.scan_parquet(hm / "processed" / ds / f"{split}_for_inference.parquet")
            .filter(~pl.col(f"{tt}_past"))
            .select(
                *pt_cols,
                pl.col("tokens_past").list.len().alias("n_tokens_first_24h"),
                f"{tt}_future",
            )
            for split in ("train", "tuning", "held_out")
        )
        cat_cols = [c for c, t in lf_train.collect_schema().items() if t == pl.String]
        cats = {
            c: sorted(
                pl.concat(
                    [lf.select(pl.col(c).unique()) for lf in (lf_train, lf_tuning)]
                )
                .collect()
                .get_column(c)
                .drop_nulls()
                .unique()
                .to_list()
            )
            for c in cat_cols
        }
        df_train, df_tuning, df_held_out = (
            lf.with_columns(
                pl.col(c).cast(pl.Enum(cats[c]), strict=False) for c in cat_cols
            )
            .collect()
            .to_pandas()
            for lf in (lf_train, lf_tuning, lf_held_out)
        )
        mdl = lgb.LGBMClassifier(n_jobs=-1)
        mdl.fit(
            X=df_train.drop(columns=f"{tt}_future"),
            y=df_train[f"{tt}_future"].astype(int),
            eval_set=[
                (
                    df_tuning.drop(columns=f"{tt}_future"),
                    df_tuning[f"{tt}_future"].astype(int),
                )
            ],
            categorical_feature=cat_cols,
            eval_metric="auc",
            callbacks=[lgb.early_stopping(stopping_rounds=25)],
        )
        yt = df_held_out[f"{tt}_future"].astype(int)
        ys = mdl.predict_proba(df_held_out.drop(columns=f"{tt}_future"))[:, 1]
        bl_roc_auc.loc[tt, ds] = skl_mets.roc_auc_score(yt, ys)
        precs, recs, _ = skl_mets.precision_recall_curve(
            yt, np.round(ys, decimals=4), drop_intermediate=True
        )
        bl_pr_auc.loc[tt, ds] = skl_mets.auc(recs, precs)


bl2_roc_auc = pd.DataFrame(index=grokked_outcome_tokens, columns=dsets)
bl2_pr_auc = bl2_roc_auc.copy()

for ds in dsets:
    lf_train, lf_tuning, lf_held_out = (
        pl.scan_parquet(
            hm / "processed" / ds / f"{split}_for_inference.parquet"
        ).with_columns(
            pl.col("tokens_past").list.len().alias("n_tokens_first_24h"),
            *[
                pl.col("tokens_past").list.count_matches(v).alias(f"count_{v}")
                for k, v in vocab.items()
            ],
        )
        for split in ("train", "tuning", "held_out")
    )
    for tt in grokked_outcome_tokens:
        df_train, df_tuning, df_held_out = (
            lf.filter(~pl.col(f"{tt}_past"))
            .select(
                "age_at_admission",
                "n_tokens_first_24h",
                *[f"count_{v}" for v in vocab.values()],
                f"{tt}_future",
            )
            .collect()
            .to_pandas()
            for lf in (lf_train, lf_tuning, lf_held_out)
        )
        mdl = lgb.LGBMClassifier(
            n_jobs=-1, callbacks=[lgb.early_stopping(stopping_rounds=25)]
        )
        mdl.fit(
            X=df_train.drop(columns=f"{tt}_future"),
            y=df_train[f"{tt}_future"].astype(int),
            eval_set=[
                (
                    df_tuning.drop(columns=f"{tt}_future"),
                    df_tuning[f"{tt}_future"].astype(int),
                )
            ],
            eval_metric="auc",
        )
        yt = df_held_out[f"{tt}_future"].astype(int)
        ys = mdl.predict_proba(df_held_out.drop(columns=f"{tt}_future"))[:, 1]
        bl2_roc_auc.loc[tt, ds] = skl_mets.roc_auc_score(yt, ys)
        precs, recs, _ = skl_mets.precision_recall_curve(
            yt, np.round(ys, decimals=4), drop_intermediate=True
        )
        bl2_pr_auc.loc[tt, ds] = skl_mets.auc(recs, precs)
