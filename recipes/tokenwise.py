#!/usr/bin/env python3

"""
load results from postprocessing.py and baselines.py and create tables
"""

import os
import pathlib

import numpy as np
import pandas as pd

pd.options.display.float_format = "{:,.3f}".format
pd.options.display.max_columns = None
pd.options.display.max_rows = 100
pd.options.display.width = None
pd.options.display.expand_frame_repr = False
pd.options.display.show_dimensions = True

hm = (
    pathlib.Path("/gpfs/data" if os.uname().nodename.startswith("cri") else "/mnt")
    / "bbj-lab/users/burkh4rt"
)


def extract_site(s):
    return (
        "UCMC"
        if "ucmc" in s.lower()
        else "NU"
        if "nu" in s.lower()
        else "MIMIC"
        if "mimic" in s.lower()
        else "ALL"
        if "all" in s.lower()
        else s.split("mdl-")[-1].upper()
    )


for metric in ("roc", "pr"):
    bl_lr = (
        pd.read_csv(hm / f"bl-{metric}-auc-LR-1H.csv", index_col=[0, 1])
        .rename_axis(index={"train": "training"})
        .rename(index=lambda s: s.removesuffix("-icu").upper(), level="training")
        .assign(model="LR")
        .set_index("model", append=True)
        .reorder_levels(["token", "model", "training"])
    )

    bl_lgbm = (
        pd.read_csv(hm / f"bl-{metric}-auc-LGBM-C.csv", index_col=[0, 1])
        .rename_axis(index={"train": "training"})
        .rename(index=lambda s: s.removesuffix("-icu").upper(), level="training")
        .assign(model="LGBM")
        .set_index("model", append=True)
        .reorder_levels(["token", "model", "training"])
    )

    tkwz = (
        pd.read_csv(hm / f"tkwz-{metric}.csv", index_col=[0, 1])
        .rename_axis(index={"models": "training"})
        .assign(
            model=lambda df: df.index.get_level_values("training").map(
                lambda s: "GEM" if "cx" not in s else "GEM-*"
            )
        )
        .rename(index=lambda s: extract_site(s), level="training")
        .set_index("model", append=True)
        .reorder_levels(["token", "model", "training"])
    )

    met = pd.concat([bl_lr, bl_lgbm, tkwz]).rename(
        columns=lambda s: s.split("-icu")[0].upper()
    )

    outcomes = {
        k.split("//")[-1]: k for k in set(met.index.get_level_values("token").tolist())
    }
    outcomes["vasopressors"] = outcomes["pressor_init"]
    del outcomes["pressor_init"]

    # relabel the token level with the short outcome names, e.g.
    # `LABEL//pressor_init` -> `vasopressors`
    met = met.rename(index={v: k for k, v in outcomes.items()}, level="token")

    # group rows by training set, ordering models explicitly within each group
    training_order = [
        "UCMC",
        "NU",
        "MIMIC",
        "C-FEDAVG10",
        "C-FEDAVGM10",
        "C-FEDADAM10",
        "ALL",
    ]
    model_order = ["LR", "LGBM", "GEM", "GEM-*"]
    rows = met.index.droplevel("token").unique().reorder_levels(["training", "model"])
    rows = rows[
        np.lexsort(
            (
                rows.get_level_values("model").map(model_order.index),
                rows.get_level_values("training").map(training_order.index),
            )
        )
    ]

    for i, part in enumerate(np.split(np.array(sorted(outcomes.keys())), 4)):
        met_by_token = (
            met.loc[lambda df: df.index.get_level_values("token").isin(part)]
            .rename_axis(columns="evaluation")
            .unstack("token")
            .swaplevel(axis="columns")
            .reorder_levels(["training", "model"])
            .reindex(
                index=rows,
                columns=pd.MultiIndex.from_product(
                    [part, met.columns], names=["token", "evaluation"]
                ),
            )
        )
        print(met_by_token.to_latex(float_format="%.3f"))

    met_by_token.to_csv(hm / f"{metric}-by-token.csv")
