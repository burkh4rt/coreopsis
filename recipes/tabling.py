#!/usr/bin/env python3

"""
load and select tabular results
"""

import pathlib

import numpy as np
import pandas as pd

pd.options.display.float_format = "{:,.3f}".format
pd.options.display.max_columns = None
pd.options.display.max_rows = 100
pd.options.display.width = None
pd.options.display.expand_frame_repr = False
pd.options.display.show_dimensions = True

hm = pathlib.Path("~/Downloads").expanduser().resolve()


bl_roc_lr = (
    pd.read_csv(hm / "bl-roc-auc-LR-1H.csv", index_col=[0, 1])
    .rename_axis(index={"train": "training"})
    .rename(index=lambda s: s.removesuffix("-icu").upper(), level="training")
    .assign(model="LR")
    .set_index("model", append=True)
    .reorder_levels(["token", "model", "training"])
)

bl_roc_lgbm = (
    pd.read_csv(hm / "bl-roc-auc-LGBM-C.csv", index_col=[0, 1])
    .rename_axis(index={"train": "training"})
    .rename(index=lambda s: s.removesuffix("-icu").upper(), level="training")
    .assign(model="LGBM")
    .set_index("model", append=True)
    .reorder_levels(["token", "model", "training"])
)

tkwz_roc = (
    pd.read_csv(hm / "tkwz-roc.csv", index_col=[0, 1])
    .rename_axis(index={"models": "training"})
    .rename(index=lambda s: s, level="training")
    .assign(
        model=lambda df: df.index.get_level_values("training").map(
            lambda s: "GEM" if "cx" not in s else "GEM-*"
        )
    )
    .set_index("model", append=True)
    .reorder_levels(["token", "model", "training"])
)

roc = pd.concat([bl_roc_lr, bl_roc_lgbm, tkwz_roc]).rename(
    columns=lambda s: s.split("-icu")[0]
)

outcomes = {
    k.split("//")[-1]: k for k in set(roc.index.get_level_values("token").tolist())
}
outcomes["vasopressors"] = outcomes["pressor_init"]
del outcomes["pressor_init"]

# relabel the token level with the short outcome names, e.g.
# `LABEL//pressor_init` -> `vasopressors`
roc = roc.rename(index={v: k for k, v in outcomes.items()}, level="token")


for i, part in enumerate(np.split(np.array(sorted(outcomes.keys())), 3)):
    roc_by_token = (
        roc.loc[lambda df: df.index.get_level_values("token").isin(part)]
        .rename_axis(columns="evaluation")
        .unstack("token")
        .swaplevel(axis="columns")
        .reindex(
            index=roc.index.droplevel("token").unique(),
            columns=pd.MultiIndex.from_product(
                [part, roc.columns], names=["token", "evaluation"]
            ),
        )
    )
    print(roc_by_token.to_latex(float_format="%.3f"))


tkwz_pr = pd.read_csv(hm / "tkwz-pr.csv")
