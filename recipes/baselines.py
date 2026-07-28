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

dsets = ["ucmc-icu", "nu-icu", "mimic-icu"]

vocab = OmegaConf.load(hm / "processed" / dsets[0] / "tokenizer.yaml").lookup

grokked_outcome_tokens = [
    x
    for x in vocab.keys()
    if any(
        fnmatch.fnmatch(x, p)
        for p in OmegaConf.load(resources.files("coreopsis.config") / "scoring.yaml")[
            "tokens_of_interest"
        ]
    )
]

bl_roc_auc = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        (grokked_outcome_tokens, dsets + ["all"]), names=("token", "train")
    ),
    columns=dsets,
)
bl_pr_auc = bl_roc_auc.copy()

df_train, df_tuning, df_held_out, mdls = dict(), dict(), dict(), dict()

for ds in dsets:
    lf_train, lf_tuning, lf_held_out = (
        pl.scan_parquet(
            hm / "processed" / ds / f"{split}_for_inference.parquet"
        ).with_columns(
            pl.col("tokens_past").list.len().alias("n_tokens_first_24h"),
            *[
                pl.col("tokens_past").list.count_matches(v).alias(f"count_{v}")
                for v in vocab.values()
            ],
        )
        for split in ("train", "tuning", "held_out")
    )
    for tt in grokked_outcome_tokens:
        df_train[(tt, ds)], df_tuning[(tt, ds)], df_held_out[(tt, ds)] = (
            lf.filter(~pl.col(f"{tt}_past"))
            .select(
                # "age_at_admission",
                # "n_tokens_first_24h",
                *[f"count_{v}" for v in vocab.values()],
                f"{tt}_future",
            )
            .collect()
            .to_pandas()
            for lf in (lf_train, lf_tuning, lf_held_out)
        )
        mdls[(tt, ds)] = lgb.LGBMClassifier(n_jobs=-1)
        mdls[(tt, ds)].fit(
            X=df_train[(tt, ds)].drop(columns=f"{tt}_future"),
            y=df_train[(tt, ds)][f"{tt}_future"].astype(int),
            eval_set=[
                (
                    df_tuning[(tt, ds)].drop(columns=f"{tt}_future"),
                    df_tuning[(tt, ds)][f"{tt}_future"].astype(int),
                )
            ],
            eval_metric="auc",
        )

for tt in grokked_outcome_tokens:
    df_train_all = pd.concat(df_train[(tt, ds)] for ds in dsets)
    df_tuning_all = pd.concat(df_tuning[(tt, ds)] for ds in dsets)
    mdls[(tt, "all")] = lgb.LGBMClassifier(n_jobs=-1)
    mdls[(tt, "all")].fit(
        X=df_train_all.drop(columns=f"{tt}_future"),
        y=df_train_all[f"{tt}_future"].astype(int),
        eval_set=[
            (
                df_tuning_all.drop(columns=f"{tt}_future"),
                df_tuning_all[f"{tt}_future"].astype(int),
            )
        ],
        eval_metric="auc",
    )

for ds_train in dsets + ["all"]:
    for ds_test in dsets:
        for tt in grokked_outcome_tokens:
            yt = df_held_out[(tt, ds_test)][f"{tt}_future"].astype(int)
            ys = mdls[(tt, ds_train)].predict_proba(
                df_held_out[(tt, ds_test)].drop(columns=f"{tt}_future")
            )[:, 1]
            bl_roc_auc.loc[(tt, ds_train), ds_test] = skl_mets.roc_auc_score(yt, ys)
            precs, recs, _ = skl_mets.precision_recall_curve(
                yt, np.round(ys, decimals=4), drop_intermediate=True
            )
            bl_pr_auc.loc[(tt, ds_train), ds_test] = skl_mets.auc(recs, precs)

print(bl_roc_auc)
print(bl_pr_auc)
