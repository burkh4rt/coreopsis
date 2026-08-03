#!/usr/bin/env python3

"""
load results from postprocessing.py and create figures
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
    # primary
    "maroon": "#800000",
    "light_greystone": "#D9D9D9",
    "greystone": "#A6A6A6",
    "dark_greystone": "#737373",
    "white": "#FFFFFF",
    "black": "#000000",
    # secondary - base shades
    "goldenrod": "#EAAA00",
    "terracotta": "#DE7C00",
    "ivy": "#789D4A",
    "forest": "#275D38",
    "lake": "#007396",
    "violet": "#59315F",
    "brick": "#A4343A",
    # secondary - light shades
    "goldenrod_light": "#F3D03E",
    "terracotta_light": "#ECA154",
    "ivy_light": "#A9C47F",
    "forest_light": "#9CAF88",
    "lake_light": "#3EB1C8",
    "violet_light": "#86647A",
    "brick_light": "#B46A55",
    # secondary - dark shades
    "goldenrod_dark": "#CC8A00",
    "terracotta_dark": "#A9431E",
    "ivy_dark": "#13301C",
    "forest_dark": "#284734",
    "lake_dark": "#002A3A",
    "violet_dark": "#41273B",
    "brick_dark": "#643335",
}

# dataset -> (display name, fedavg model trained on the *other two* sites,
#             full training-set size)
frac_dsets = {
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
# training-set size, so no panel's axis implies more data than the site has,
# and the panel widths are set proportionally (see column_widths below)
x_lo = min(fracs) * min(size for *_, size in frac_dsets.values())

# shared, uniform ticks across all panels (plotly clips to each panel's range);
# one label per decade-ish keeps them horizontal in the narrow side-by-side panels
x_tickvals = (200, 1000, 5000, 20000)
x_ticktext = ("200", "1k", "5k", "20k")

# number of federation rounds swept during training -> model suffix
fed_rounds = (1, 5, 10, 50)

# dataset -> (display name, color); per-dataset palette, independent of the
# role-based colors used by the data-fraction figure
rnds_dsets = {
    "ucmc-icu": ("UCMC", colors["lake"]),
    "nu-icu": ("NU", colors["terracotta"]),
    "mimic-icu": ("MIMIC", colors["forest"]),
}

# postprocessing writes one bootstrap-CI table per experiment ('{exp}-{roc,pr}.csv');
# each figure below reads only the table for the experiment that produced its
# models, so every point within a figure comes from the same bootstrap run
nan_ci = (float("nan"), float("nan"))

missing = []  # (metric, model, dataset) triples absent from the CI tables


def rgba(hex_color, alpha):
    """Convert '#RRGGBB' to 'rgba(r, g, b, alpha)' for translucent fills."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


def load_cis(csv_name):
    """Read a CI table (cells are numpy '[lo hi]' reprs, rows are models,
    columns are datasets) into a DataFrame of (lo, hi) float tuples."""

    def parse(s):
        if pd.isna(s):
            return nan_ci
        lo, hi = (float(x) for x in str(s).strip("[]").split())
        return (lo, hi)

    return pd.read_csv(hm / csv_name, index_col=0).rename_axis("models").map(parse)


def get_ci(ci, mdl, ds, metric_slug):
    """CI for one model × dataset; models that have not been scored yet plot
    as gaps and are reported at the end."""
    if mdl in ci.index:
        lo, hi = ci.loc[mdl, ds]
        if not (math.isnan(lo) or math.isnan(hi)):
            return (lo, hi)
    missing.append((metric_slug, mdl, ds))
    return nan_ci


def center(ci_tuple):
    """Point estimate for a CI: its midpoint, matching how the manuscript
    tables are formatted by postprocessing.re_fmt_ci."""
    lo, hi = ci_tuple
    return (lo + hi) / 2


def make_error_y(cis, color, s):
    """Asymmetric plotly error bars from (lo, hi) CIs, measured off each
    plotted center."""
    return dict(
        type="data",
        symmetric=False,
        array=[hi - center((lo, hi)) for lo, hi in cis],
        arrayminus=[center((lo, hi)) - lo for lo, hi in cis],
        color=color,
        thickness=2 * s,  # matches the line width of the trace it annotates
        width=6 * s * 0.6,  # cap width: 60% of the marker size
    )


for metric, metric_label, metric_slug in (
    ("roc", "ROC-AUC", "roc-auc"),
    ("pr", "PR-AUC", "pr-auc"),
):
    """
    performance vs. fraction of a site's training data, with fed baselines
    """
    # frac-* carries the fraction sweep plus the fed / pooled baselines it is
    # compared against
    ci = load_cis(f"frac-{metric}.csv")

    # every text and mark element is sized against a *reference* width -- 900px,
    # the width at which the base sizes apply (scale factor 1.0) -- and not
    # against fig_width itself. The figures are rescaled to a common width in the
    # manuscript, so scaling the type with the canvas would make widening a
    # figure a no-op: panels and text would grow together and the rescaled result
    # would be identical. Pinning s and exporting a wider canvas is what actually
    # buys the panels room, at the cost of type that reads smaller once rescaled.
    fig_width = 1350  # 50% wider than the 900px reference
    s = 1.0  # scale factor for all sized elements

    n_cols = len(frac_dsets)

    # panels side by side on one shared y axis; each panel's width is
    # proportional to the log-span up to its own site's training-set size, so
    # every panel shares a single log x-scale
    fig = make_subplots(
        rows=1,
        cols=n_cols,
        shared_yaxes=True,
        subplot_titles=[name for name, *_ in frac_dsets.values()],
        horizontal_spacing=0.05,  # gap keeps neighboring panels' tick labels apart
        column_widths=[
            math.log10(size) - math.log10(x_lo) for *_, size in frac_dsets.values()
        ],
    )
    fig.update_annotations(font_size=36 * s)  # subplot (panel) titles

    y_extremes = []  # every CI bound plotted, over all panels -> shared y range

    for col, (ds, (name, other_fed, size)) in enumerate(frac_dsets.items(), start=1):
        show = col == 1  # one shared legend

        # approximate training-set size at each fraction of this site's data
        sizes = [f * size for f in fracs]

        # horizontal federated / pooled baselines (model looked up per site),
        # each with its role-based color *and* its own dash pattern, so the three
        # stay tellable apart in grayscale print and for red-green color vision
        # deficiency (ivy vs. brick is exactly the confusable pair). The dash
        # period grows with how much data the model saw -- dotted for the two
        # held-out sites, dashed for all three federated, long-dashed for pooled
        # -- and the solid, marker-carrying sweep is distinct from all three.
        baselines = [
            ("fedavg (other two sites)", other_fed, colors["ivy"], "dot"),
            ("fedavg (all three sites)", "mdl-c-fedavg10", colors["brick"], "dash"),
            ("all data (pooled)", "mdl-c-all", colors["goldenrod_dark"], "longdash"),
        ]

        # CI bands for the baselines, drawn behind everything else; these models
        # see fixed amounts of data, so they span this panel's whole x-axis
        # rather than only the extent of the site's sweep
        for label, mdl, color, _ in baselines:
            lo, hi = get_ci(ci, mdl, ds, metric_slug)
            y_extremes += [lo, hi]
            fig.add_trace(
                go.Scatter(
                    x=[x_lo, size, size, x_lo],
                    y=[lo, lo, hi, hi],
                    fill="toself",
                    fillcolor=rgba(color, 0.06),  # translucent band fill
                    mode="lines",
                    line=dict(width=0),
                    legendgroup=label,
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=col,
            )

        # baseline center lines drawn on top of their CI bands
        for label, mdl, color, dash in baselines:
            val = center(get_ci(ci, mdl, ds, metric_slug))
            fig.add_trace(
                go.Scatter(
                    x=[x_lo, size],
                    y=[val, val],
                    mode="lines",
                    name=label,
                    legendgroup=label,
                    showlegend=show,
                    line=dict(color=color, width=2 * s, dash=dash),
                ),
                row=1,
                col=col,
            )

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
                line=dict(color=colors["dark_greystone"], width=2 * s),
                marker=dict(size=6 * s, color=colors["dark_greystone"]),
                error_y=make_error_y(curve_ci, colors["dark_greystone"], s),
            ),
            row=1,
            col=col,
        )

        # the axis stops exactly at this site's full training size; combined with
        # the column_widths above, every panel then reads on one log x-scale
        fig.update_xaxes(
            type="log",
            range=[math.log10(x_lo), math.log10(size)],
            tickvals=x_tickvals,
            ticktext=x_ticktext,
            tickangle=0,
            row=1,
            col=col,
        )

    # label only the middle panel, so the one title reads as covering the row
    fig.update_xaxes(title_text="training size", row=1, col=(n_cols + 1) // 2)

    # shared_yaxes already ties the panels to a common scale; the explicit range
    # pads it so the sweep's end-point markers and CI caps clear the axis edges
    y_bounds = [v for v in y_extremes if not math.isnan(v)]
    y_pad = 0.06 * (max(y_bounds) - min(y_bounds))
    fig.update_yaxes(range=[min(y_bounds) - y_pad, max(y_bounds) + y_pad])
    fig.update_yaxes(title_text=f"mean {metric_label}", row=1, col=1)
    fig.update_layout(
        template="plotly_white",
        font=dict(
            # uchicago brand face; base size covers axis titles, tick labels,
            # and legend entries
            family="Gotham Book, Gotham, Helvetica, sans-serif",
            size=30 * s,
            color="black",
        ),
        title=dict(
            text=f"Mean {metric_label} vs. training set size",
            font=dict(size=42 * s, color="black"),
        ),
        # legend beside the panels rather than under them; these entry labels are
        # long, so it gets a smaller size than the base font to leave the three
        # panels usable width (plotly's margin autoexpand makes room for it)
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02,
            font=dict(size=22 * s),
            # wider than plotly's 30px default so each entry's swatch is long
            # enough to show a full dash period -- at the default, 'longdash'
            # draws as one segment and reads as solid
            itemwidth=60,
        ),
        margin=dict(l=60, r=30, t=150, b=110),
        width=fig_width,
        height=600,
    )
    fig.write_image(hm / f"data-fraction-sweep-{metric_slug}-aggregate.pdf")

    """
    average performance vs. number of federation rounds (all-site fedavg)
    """
    # one line per dataset, metric aggregated over outcome tokens
    ci = load_cis(f"rnds-{metric}.csv")

    fig_width = 975  # 50% wider than the 650px this panel's type is sized for
    s = 650 / 900  # scale factor for all sized elements (see above)

    fig = go.Figure()
    for ds, (name, color) in rnds_dsets.items():
        curve_ci = [get_ci(ci, f"mdl-c-fedavg{n}", ds, metric_slug) for n in fed_rounds]
        fig.add_trace(
            go.Scatter(
                x=fed_rounds,
                y=[center(c) for c in curve_ci],
                mode="lines+markers",
                name=name,
                line=dict(color=color, width=2 * s),
                marker=dict(size=6 * s, color=color),
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
        font=dict(
            family="Gotham Book, Gotham, Helvetica, sans-serif",
            size=30 * s,
            color="black",
        ),
        title=dict(
            text=f"Mean {metric_label} vs. federation rounds",
            font=dict(size=42 * s, color="black"),
        ),
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02),
        margin=dict(l=60, r=30, t=60, b=100),
        width=fig_width,
        height=500,
    )
    fig.write_image(hm / f"fed-rounds-sweep-{metric_slug}.pdf")

if missing:
    print(f"\n{len(missing)} model × dataset CIs missing (plotted as gaps):")
    for metric_slug, mdl, ds in sorted(set(missing)):
        print(f"  {metric_slug}: {mdl} × {ds}")
