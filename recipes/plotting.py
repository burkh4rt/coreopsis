#!/usr/bin/env python3

"""
plot results
"""

import math
import pathlib

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

pd.options.display.float_format = "{:,.3f}".format
pd.options.display.max_columns = None
pd.options.display.max_rows = 100
pd.options.display.width = None
pd.options.display.expand_frame_repr = False
pd.options.display.show_dimensions = True

hm = pathlib.Path("~/Downloads").expanduser().resolve()

colors = {
    # Primary
    "maroon": "#800000",
    "light_greystone": "#D9D9D9",
    "greystone": "#A6A6A6",
    "dark_greystone": "#737373",
    "white": "#FFFFFF",
    "black": "#000000",
    # Secondary - base shades
    "goldenrod": "#EAAA00",
    "terracotta": "#DE7C00",
    "ivy": "#789D4A",
    "forest": "#275D38",
    "lake": "#007396",
    "violet": "#59315F",
    "brick": "#A4343A",
    # Secondary - light shades
    "goldenrod_light": "#F3D03E",
    "terracotta_light": "#ECA154",
    "ivy_light": "#A9C47F",
    "forest_light": "#9CAF88",
    "lake_light": "#3EB1C8",
    "violet_light": "#86647A",
    "brick_light": "#B46A55",
    # Secondary - dark shades
    "goldenrod_dark": "#CC8A00",
    "terracotta_dark": "#A9431E",
    "ivy_dark": "#13301C",
    "forest_dark": "#284734",
    "lake_dark": "#002A3A",
    "violet_dark": "#41273B",
    "brick_dark": "#643335",
}

# ---------------------------------------------------------------------------
# shared style / configuration (metric-independent)
# ---------------------------------------------------------------------------
# validated categorical palette (light mode)
COL_CURVE = colors["dark_greystone"]  #  site-trained sweep
COL_OTHER = colors["ivy"]  # fedavg on the other two sites
COL_FED = colors["brick"]  # fedavg on all three sites
COL_ALL = colors["goldenrod_dark"]  # single model pooled over all data

# uniform label/mark sizing after rescaling:
# each figure is exported at its own pixel width (multi-panel vs. single panel)
# but rescaled to a common width in the manuscript. Sizing every text and mark
# element proportionally to the figure's own width makes labels, ticks, markers,
# and lines appear uniform once all figures share the same final width.
REF_WIDTH = 900  # width at which the base sizes below apply (scale factor 1.0)
FONT_FAMILY = "Gotham Book, Gotham, Helvetica, sans-serif"  # UChicago brand face
FONT_SIZE = 30  # axis titles, tick labels, legend entries
TITLE_SIZE = 42  # main figure title
SUBTITLE_SIZE = 36  # subplot (panel) titles
LINE_WIDTH = 2
MARKER_SIZE = 6
BAND_ALPHA = 0.06  # translucent fill for the baseline confidence bands

# dataset -> (display name, fedavg model trained on the *other two* sites,
#             full training-set size)
plot_dsets = {
    "ucmc-icu": ("UCMC", "mdl-c-fedavg10-mn", 15399),  # other two = mimic + nu
    "nu-icu": ("NU", "mdl-c-fedavg10-mc", 46030),  # other two = mimic + ucmc
    "mimic-icu": ("MIMIC", "mdl-c-fedavg10-cn", 24146),  # other two = ucmc + nu
}

# numerators (out of 100) of the fractions of each site's data that were used:
# 1..10 percent, then 20, ..., 100 percent -> model suffix is the 3-digit value
frac_nums = list(range(1, 11)) + list(range(20, 110, 10))
fracs = [n / 100 for n in frac_nums]

# common lower bound for the (log) x-axes: smallest sample count plotted
# anywhere. Each panel's *upper* bound is exactly its own site's full
# training-set size, so no panel's axis implies more data than the site has;
# x_hi, the largest of those, is the panel that gets the full figure width
x_lo = min(fracs) * min(size for *_, size in plot_dsets.values())
x_hi = max(size for *_, size in plot_dsets.values())
x_log_span = math.log10(x_hi) - math.log10(x_lo)

# shared, uniform ticks across all panels (plotly clips to each panel's range)
x_tickvals = [200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
x_ticktext = ["200", "500", "1k", "2k", "5k", "10k", "20k", "50k"]

# number of federation rounds swept during training -> model suffix
fed_rounds = [1, 5, 10, 50]

# dataset -> (display name, color); per-dataset palette, independent of the
# role-based palette used by the data-fraction plots above
round_dsets = {
    "ucmc-icu": ("UCMC", colors["lake"]),  # lake
    "nu-icu": ("NU", colors["terracotta"]),  # terracotta
    "mimic-icu": ("MIMIC", colors["forest"]),  # forest
}

# postprocessing writes one bootstrap-CI table per experiment ('{exp}-{roc,pr}.csv');
# each figure below reads only the table for the experiment that produced its
# models, so every point within a figure comes from the same bootstrap run
NAN_CI = (float("nan"), float("nan"))

missing = []  # (metric, model, dataset) triples absent from the CI tables


def rgba(hex_color, alpha):
    """'#RRGGBB' -> 'rgba(r, g, b, alpha)' for translucent fills."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


def load_cis(csv_name):
    """Bootstrap 95% CIs saved as '[lo hi]' numpy-repr strings, indexed by
    model with one column per dataset -> DataFrame of (lo, hi) float tuples."""

    def parse(s):
        if pd.isna(s):
            return NAN_CI
        lo, hi = (float(x) for x in str(s).strip("[]").split())
        return (lo, hi)

    return pd.read_csv(hm / csv_name, index_col=0).rename_axis("models").map(parse)


def get_ci(ci, mdl, ds, metric_slug):
    """CI tuple for one model × dataset, tolerating models that have not been
    scored yet: those plot as gaps and are reported at the end."""
    if mdl in ci.index:
        lo, hi = ci.loc[mdl, ds]
        if not (math.isnan(lo) or math.isnan(hi)):
            return (lo, hi)
    missing.append((metric_slug, mdl, ds))
    return NAN_CI


def center(ci_tuple):
    """point estimate = midpoint of the bootstrap CI (matches how the tables
    in the manuscript are formatted by postprocessing.re_fmt_ci)"""
    lo, hi = ci_tuple
    return (lo + hi) / 2


def make_error_y(cis, color, s):
    """asymmetric plotly error bars from bootstrap (lo, hi) CI tuples,
    measured off each plotted center (the CI midpoint)."""
    return dict(
        type="data",
        symmetric=False,
        array=[hi - center((lo, hi)) for lo, hi in cis],
        arrayminus=[center((lo, hi)) - lo for lo, hi in cis],
        color=color,
        thickness=LINE_WIDTH * s,
        width=MARKER_SIZE * s * 0.6,
    )


# metric -> (CI csv suffix, axis/title label, output slug)
metrics = [("roc", "ROC-AUC", "roc-auc"), ("pr", "PR-AUC", "pr-auc")]

for metric_key, metric_label, metric_slug in metrics:
    # -----------------------------------------------------------------------
    # performance vs. fraction of a site's training data, with fed baselines
    # -----------------------------------------------------------------------
    # frac-* carries the fraction sweep plus the fed / pooled baselines it is
    # compared against
    ci = load_cis(f"frac-{metric_key}.csv")

    fig_width = 650
    s = fig_width / REF_WIDTH  # scale factor for all sized elements

    n_rows = len(plot_dsets)

    # each panel keeps its own x axis (they end at different training sizes), so
    # no shared_xaxes here; the spacing leaves room for per-panel tick labels
    fig = make_subplots(
        rows=n_rows,
        cols=1,
        subplot_titles=[name for name, *_ in plot_dsets.values()],
        vertical_spacing=0.10,
    )
    # subplot titles, set flush with the panels' left edge instead of centered
    # (every panel's x domain starts at the same left edge)
    fig.update_annotations(
        font_size=SUBTITLE_SIZE * s, x=fig.layout.xaxis.domain[0], xanchor="left"
    )

    y_extremes = []  # every CI bound plotted, over all panels -> shared y range

    for row, (ds, (name, other_fed, size)) in enumerate(plot_dsets.items(), start=1):
        show = row == 1  # one shared legend

        # approximate training-set size at each fraction of this site's data
        sizes = [f * size for f in fracs]

        # horizontal federated / pooled baselines (model looked up per site)
        baselines = [
            ("fedavg (other two sites)", other_fed, COL_OTHER),
            ("fedavg (all three sites)", "mdl-c-fedavg10", COL_FED),
            ("all data (pooled)", "mdl-c-all", COL_ALL),
        ]

        # 95% CI bands for the baselines, drawn behind everything else; these
        # models see fixed amounts of data, so they span this panel's whole
        # x-axis rather than only the extent of the site's sweep
        for label, mdl, color in baselines:
            lo, hi = get_ci(ci, mdl, ds, metric_slug)
            y_extremes += [lo, hi]
            fig.add_trace(
                go.Scatter(
                    x=[x_lo, size, size, x_lo],
                    y=[lo, lo, hi, hi],
                    fill="toself",
                    fillcolor=rgba(color, BAND_ALPHA),
                    mode="lines",
                    line=dict(width=0),
                    legendgroup=label,
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=1,
            )

        # baseline center lines drawn on top of their CI bands
        for label, mdl, color in baselines:
            val = center(get_ci(ci, mdl, ds, metric_slug))
            fig.add_trace(
                go.Scatter(
                    x=[x_lo, size],
                    y=[val, val],
                    mode="lines",
                    name=label,
                    legendgroup=label,
                    showlegend=show,
                    line=dict(color=color, width=LINE_WIDTH * s, dash="dash"),
                ),
                row=row,
                col=1,
            )

        # site-trained sweep over increasing fractions of this site's data, with
        # asymmetric 95% CI error bars. Added last so it paints over the
        # baselines; legendrank keeps it first in the legend anyway. The sweep's
        # end points sit exactly on the axis bounds, so cliponaxis lets their
        # markers and error-bar caps spill past the edge instead of being clipped
        curve_ci = [
            get_ci(ci, f"mdl-c-{ds}-{n:03d}", ds, metric_slug) for n in frac_nums
        ]
        y_extremes += [b for c in curve_ci for b in c]
        fig.add_trace(
            go.Scatter(
                x=sizes,
                y=[center(c) for c in curve_ci],
                mode="lines+markers",
                name="site-trained",
                legendgroup="site-trained",
                legendrank=0,
                showlegend=show,
                cliponaxis=False,
                line=dict(color=COL_CURVE, width=LINE_WIDTH * s),
                marker=dict(size=MARKER_SIZE * s, color=COL_CURVE),
                error_y=make_error_y(curve_ci, COL_CURVE, s),
            ),
            row=row,
            col=1,
        )

        # the axis stops exactly at this site's full training size, and the panel
        # is narrowed to its share of the largest site's log-span so that all
        # panels still share one log x-scale (the vertical-stack counterpart of
        # the side-by-side layout's column_widths)
        fig.update_xaxes(
            type="log",
            range=[math.log10(x_lo), math.log10(size)],
            domain=[0, (math.log10(size) - math.log10(x_lo)) / x_log_span],
            tickvals=x_tickvals,
            ticktext=x_ticktext,
            row=row,
            col=1,
        )

    fig.update_xaxes(title_text="training size", row=n_rows, col=1)

    # stacked panels each own their y axis (make_subplots' shared_yaxes only
    # shares across columns), so the comparability the side-by-side version got
    # for free comes from one explicit range spanning every panel's CIs
    y_bounds = [v for v in y_extremes if not math.isnan(v)]
    y_pad = 0.06 * (max(y_bounds) - min(y_bounds))
    fig.update_yaxes(range=[min(y_bounds) - y_pad, max(y_bounds) + y_pad])
    # label only the middle panel, so the one title reads as covering the stack
    fig.update_yaxes(title_text=f"mean {metric_label}", row=(n_rows + 1) // 2, col=1)
    fig.update_layout(
        template="plotly_white",
        font=dict(family=FONT_FAMILY, size=FONT_SIZE * s, color="black"),
        title=dict(
            text=f"Mean {metric_label} vs. training set size",
            font=dict(size=TITLE_SIZE * s, color="black"),
        ),
        legend=dict(orientation="h", yanchor="top", y=-0.10, xanchor="center", x=0.5),
        margin=dict(l=60, r=30, t=110, b=200),
        width=fig_width,
        height=1450,
    )
    fig.write_image(hm / f"data-fraction-sweep-{metric_slug}-aggregate.pdf")

    # -----------------------------------------------------------------------
    # average performance vs. number of federation rounds (all-site FedAvg)
    # -----------------------------------------------------------------------
    # one line per dataset, metric aggregated over outcome tokens
    ci = load_cis(f"rnds-{metric_key}.csv")

    fig_width = 650
    s = fig_width / REF_WIDTH  # scale factor for all sized elements

    fig = go.Figure()
    for ds, (name, color) in round_dsets.items():
        curve_ci = [get_ci(ci, f"mdl-c-fedavg{n}", ds, metric_slug) for n in fed_rounds]
        fig.add_trace(
            go.Scatter(
                x=fed_rounds,
                y=[center(c) for c in curve_ci],
                mode="lines+markers",
                name=name,
                line=dict(color=color, width=LINE_WIDTH * s),
                marker=dict(size=MARKER_SIZE * s, color=color),
                error_y=make_error_y(curve_ci, color, s),
            )
        )

    fig.update_xaxes(
        title_text="federation rounds",
        type="log",
        tickvals=fed_rounds,
        ticktext=[str(n) for n in fed_rounds],
    )
    fig.update_yaxes(title_text=f"mean {metric_label}")
    fig.update_layout(
        template="plotly_white",
        font=dict(family=FONT_FAMILY, size=FONT_SIZE * s, color="black"),
        title=dict(
            text=f"Mean {metric_label} vs. federation rounds",
            font=dict(size=TITLE_SIZE * s, color="black"),
        ),
        legend=dict(orientation="h", yanchor="top", y=-0.28, xanchor="center", x=0.5),
        margin=dict(l=60, r=30, t=60, b=150),
        width=fig_width,
        height=570,
    )
    fig.write_image(hm / f"fed-rounds-sweep-{metric_slug}.pdf")

if missing:
    print(f"\n{len(missing)} model × dataset CIs missing (plotted as gaps):")
    for metric_slug, mdl, ds in sorted(set(missing)):
        print(f"  {metric_slug}: {mdl} × {ds}")
