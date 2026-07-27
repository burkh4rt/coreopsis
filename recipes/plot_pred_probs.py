#!/usr/bin/env python3

"""
plot marginal distributions of predicted outcome probabilities.

For a model (default `mdl-all`) under generative inference, render one PDF per
method: a faceted 3-column grid of histograms, one panel per outcome token,
with the three datasets overlaid (translucent) so their predicted-probability
distributions can be compared directly. Each histogram is normalized to a
fraction (`histnorm="probability"`) so datasets of very different size (nu ~46k
vs ucmc ~15k eligible rows) are comparable in shape rather than in raw count.

Distributions are taken over the *eligible* rows -- those where the outcome has
not already occurred (~past), the same mask used throughout
recipes/postprocessing.py.

Reads the score parquet directly, so this must run where the data lives (the
cluster). PDFs land in `{hm}/plots/`.
"""

import fnmatch
import importlib.resources as resources
import math
import os
import pathlib

import numpy as np
import plotly.graph_objects as go
import polars as pl
from omegaconf import OmegaConf
from plotly.subplots import make_subplots

# --- data location (mirrors recipes/postprocessing.py) ---------------------
hm = (
    pathlib.Path("/gpfs/data" if os.uname().nodename.startswith("cri") else "/mnt")
    / "bbj-lab/users/burkh4rt"
)

dsets = ("ucmc-icu", "nu-icu", "mimic-icu")
MDL = "mdl-all"
INFERENCE = "generative"
METHODS = ("mc", "scope", "reach")

# UChicago brand palette (subset from recipes/plotting.py)
colors = {
    "maroon": "#800000",
    "light_greystone": "#D9D9D9",
    "greystone": "#A6A6A6",
    "dark_greystone": "#737373",
    "lake": "#007396",
    "terracotta": "#DE7C00",
    "forest": "#275D38",
}

# per-dataset (display name, color) -- same categorical mapping as the
# round-sweep plots in recipes/plotting.py, so a dataset keeps its color across
# figures. CVD-safe (validated: worst adjacent deltaE 18.9); the always-present
# legend is the secondary encoding, so identity never rests on color alone.
DSET_STYLE = {
    "ucmc-icu": ("UCMC", colors["lake"]),
    "nu-icu": ("NU", colors["terracotta"]),
    "mimic-icu": ("MIMIC", colors["forest"]),
}
GRID_COLOR = colors["light_greystone"]
FILL_ALPHA = 0.45  # translucent fills so overlaid datasets read through

# sizing proportional to figure width (per recipes/plotting.py), with base sizes
# tuned for a dense multi-panel grid rather than the single-panel manuscript figs
REF_WIDTH = 900
FIG_WIDTH = 1000
s = FIG_WIDTH / REF_WIDTH
MAIN_TITLE_SIZE = 24
PANEL_TITLE_SIZE = 15
AXIS_TITLE_SIZE = 15
TICK_SIZE = 11
FONT_FAMILY = "CMU Serif, Latin Modern Roman, serif"

NCOLS = 3
NBINS = 20  # bin width 0.05 over the unit interval


def rgba(hex_color: str, alpha: float) -> str:
    """'#RRGGBB' -> 'rgba(r, g, b, alpha)' for translucent fills."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


def load_outcome_tokens(home: pathlib.Path, datasets) -> list[str]:
    """Outcome tokens of interest that survive into the tokenizer (mirrors
    recipes/postprocessing.py): scoring.yaml globs intersected with the learned
    tokenizer lookup of the first dataset."""
    lookup = OmegaConf.load(home / "processed" / datasets[0] / "tokenizer.yaml").lookup
    patterns = OmegaConf.load(resources.files("coreopsis.config") / "scoring.yaml")[
        "tokens_of_interest"
    ]
    return [x for x in lookup.keys() if any(fnmatch.fnmatch(x, p) for p in patterns)]


def short(tt: str) -> str:
    """`LABEL//pressor_init` -> `pressor_init` (drop the `namespace//` prefix)."""
    return tt.split("//", 1)[-1]


def eligible_scores(df: pl.DataFrame, tt: str, method: str) -> np.ndarray:
    """Predicted probabilities for token `tt` over eligible rows (~past)."""
    past, score = (
        df.select(pl.col(f"{tt}_past"), pl.col(f"{tt}_{method}_score")).to_numpy().T
    )
    eligible = ~past.astype(bool)
    # scores are probabilities; clip any numerical spillover into [0, 1]
    return np.clip(np.nan_to_num(score.astype(float))[eligible], 0.0, 1.0)


def plot_method(
    dfs: dict[str, pl.DataFrame], method: str, tokens: list[str], out_dir: pathlib.Path
) -> pathlib.Path:
    """One faceted PDF: per outcome, the three datasets' predicted-probability
    histograms overlaid translucently."""
    nrows = math.ceil(len(tokens) / NCOLS)
    fig = make_subplots(
        rows=nrows,
        cols=NCOLS,
        subplot_titles=[short(tt) for tt in tokens],
        horizontal_spacing=0.06,
        vertical_spacing=max(0.04, 0.28 / nrows),
    )
    # subplot titles are the only annotations at this point -> size them alone,
    # before we append the shared axis-title annotations below
    for ann in fig.layout.annotations:
        ann.font.size = PANEL_TITLE_SIZE * s
        ann.font.family = FONT_FAMILY
        ann.font.color = "black"

    for i, tt in enumerate(tokens):
        row, col = divmod(i, NCOLS)
        for ds, (name, color) in DSET_STYLE.items():
            if ds not in dfs:
                continue
            fig.add_trace(
                go.Histogram(
                    x=eligible_scores(dfs[ds], tt, method),
                    xbins=dict(start=0.0, end=1.0, size=1.0 / NBINS),
                    histnorm="probability",
                    marker=dict(
                        color=rgba(color, FILL_ALPHA), line=dict(color=color, width=1)
                    ),
                    name=name,
                    legendgroup=name,
                    showlegend=(i == 0),  # one shared legend
                    hovertemplate=(
                        f"{name}<br>prob %{{x}}<br>fraction %{{y:.3f}}<extra></extra>"
                    ),
                ),
                row=row + 1,
                col=col + 1,
            )

    fig.update_xaxes(
        range=[0, 1],
        tickvals=[0, 0.5, 1],
        tickfont=dict(size=TICK_SIZE * s),
        gridcolor=GRID_COLOR,
        zeroline=False,
    )
    fig.update_yaxes(
        tickfont=dict(size=TICK_SIZE * s), gridcolor=GRID_COLOR, zeroline=False
    )

    # shared axis titles (added after the subplot-title sizing loop above)
    fig.add_annotation(
        text="predicted probability",
        x=0.5,
        y=-0.05,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=AXIS_TITLE_SIZE * s, family=FONT_FAMILY, color="black"),
    )
    fig.add_annotation(
        text="fraction of eligible patients",
        x=-0.06,
        y=0.5,
        xref="paper",
        yref="paper",
        textangle=-90,
        showarrow=False,
        font=dict(size=AXIS_TITLE_SIZE * s, family=FONT_FAMILY, color="black"),
    )

    fig.update_layout(
        template="plotly_white",
        barmode="overlay",
        bargap=0.05,
        font=dict(family=FONT_FAMILY, size=TICK_SIZE * s, color="black"),
        title=dict(
            text=f"Predicted P(outcome) — {MDL}, {INFERENCE}/{method}",
            font=dict(size=MAIN_TITLE_SIZE * s, color="black"),
            x=0.5,
            xanchor="center",
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.10,
            xanchor="center",
            x=0.5,
            font=dict(size=AXIS_TITLE_SIZE * s),
        ),
        width=FIG_WIDTH,
        height=260 * nrows + 170,
        margin=dict(l=90, r=40, t=90, b=150),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"pred-prob-{MDL}-{INFERENCE}-{method}.pdf"
    fig.write_image(out)
    return out


if __name__ == "__main__":
    tokens = load_outcome_tokens(hm, dsets)
    out_dir = hm / "plots"
    needed = [
        c
        for tt in tokens
        for c in (f"{tt}_past", *(f"{tt}_{m}_score" for m in METHODS))
    ]
    dfs = {}
    for ds in dsets:
        try:
            dfs[ds] = (
                pl.scan_parquet(
                    hm / "processed" / ds / MDL / f"scores-{INFERENCE}-*.parquet"
                )
                .select(needed)
                .collect()
            )
        except (FileNotFoundError, pl.exceptions.ComputeError) as e:
            print(f"skip {ds}: {type(e).__name__}: {e}")
    for method in METHODS:
        print(f"wrote {plot_method(dfs, method, tokens, out_dir)}")
