import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from helpers import compute_speedup, idx_keep_only, idx_query, idx_query_mask


def dispatch_plot(func, case, ax_handler, append_to_fn, *args, **kwargs):
    """trys to generate a plot and writes to file based on func.__name__ and args"""
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 5), sharey=True)
    groups, args = args[0], list(args[1:])
    labels = []
    try:
        for group in groups:
            plot_properties = group.plot_properties
            plot_properties["color_cycle"] = group.color_cycle
            legend = group.name + args[0]
            fig, axes, labels = func(
                fig, axes, labels, group.df, plot_properties, legend, **kwargs
            )
        data_repository = os.environ.get("EXASIM_DATA_REPOSITORY")
        system_name = os.environ.get("EXASIM_SYSTEM_NAME")
        func_args_str = "_".join([f"{k}={v}" for k, v in kwargs.items()])
        ax_handler(axes)
        axes.legend(labels)
        fn = f"{data_repository}/{case}/figs/{system_name}/{func.__name__}_{func_args_str}_{append_to_fn}.png"
        print("save", fn)
        fig.savefig(
            Path(fn),
            bbox_inches="tight",
        )
        return fig, axes
    except Exception as e:
        print("failed to plot", func.__name__, e)
        return fig, axes


def facets_over_x(df: pd.DataFrame, legend: str, queries: dict, x: str, y: str):
    """Plot faceted by a set of queries

    Parameters:
     - legend: a string formatable by query dict key
     - queries: a dictionary of query name and a list queries
     - x: name of the index to plot over
     - y: name of the column to plot

    Returns:
     - the figure
     - the axes
    """
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 5), sharey=True)

    labels = []
    # generate over different queries

    for name, q in queries.items():
        query_mask = idx_query_mask(df, q)
        filtered_df = df[query_mask]

        # keep only x as indices to keep plot axis clean
        filtered_df = idx_keep_only(filtered_df, [x])

        filtered_df[y].plot()
        labels.append(legend.format(name))

    axes.legend(labels)

    return fig, axes


def facets_relative_to_base_over_x(
    fig,
    axes,
    labels: list,
    df: pd.DataFrame,
    plot_properties: dict,
    legend: str,
    base_query: list,
    x: str,
    y: str,
    facet: str,
):
    """faceted plot normalised values over x

    Parameters:
     - legend: a string formatable by query dict key
     - queries: a dictionary of query name and a list queries
     - x: name of the index to plot over
     - y: name of the column to plot

    Returns:
     - the figure
     - the axes
    """
    # generate over different partitionings
    # get base_ranks TODO find a generic way to do this

    base_query_mask = idx_query_mask(df, base_query)
    not_base_query_mask = np.logical_not(base_query_mask)

    # get available facets for non base case
    facet_values = set(df[not_base_query_mask].index.get_level_values(facet))

    color_func = None
    if plot_properties:
        color_func = plot_properties.pop("color_cycle", {})

    for i, facet_value in enumerate(facet_values):
        # compute individual speed up
        # pre filter DataFrame to contain either base or facet values
        filtered_df = df[
            (df.index.get_level_values(facet) == facet_value) | base_query_mask
        ]

        speedup = compute_speedup(
            filtered_df,
            base_query,
            ignore_indices=[facet],
            drop_indices=["solver"],
        )

        # remove reference values
        speedup = speedup[np.logical_not(idx_query_mask(speedup, base_query))]

        # keep only x as indices to keep plot axis clean
        speedup = idx_keep_only(speedup, [x])
        print(speedup)

        if not color_func:
            speedup[y].plot(ax=axes, **plot_properties)
        else:
            color = (
                color_func[i]
                if isinstance(color_func, list)
                else color_func[facet_value]
            )
            speedup[y].plot(ax=axes, color=color, **plot_properties)
        labels.append(legend.format(facet_value))

    return fig, axes, labels


def bar_facet(df: pd.DataFrame, facet: str):

    facet_values = list(set(df.index.get_level_values(facet)))
    facet_values.sort()

    fig, axes = plt.subplots(
        nrows=1, ncols=len(facet_values), figsize=(8, 5), sharey=True
    )

    for f, ax in zip(facet_values, axes.flatten()):
        filtered_df = df[(df.index.get_level_values(facet) == f)]

        filtered_df = filtered_df.reset_index(level=[facet], drop=True).sort_index()
        filtered_df.plot.bar(ax=ax, stacked=True, legend=False)
        ax.set_title(f"{facet} = {f}")

    h, l = ax.get_legend_handles_labels()
    l = [_.replace("_rel", "").replace(":", "") for _ in l]

    axes[-1].legend(h, l, loc="center left", bbox_to_anchor=(1.0, 0.5))
    return fig, axes


def ax_handler_wrapper(x_label=False, y_label=False, getter=lambda x: x):
    def ax_handler(axes):
        if x_label:
            getter(axes).set_xlabel(x_label)
        if y_label:
            getter(axes).set_ylabel(y_label)

    return ax_handler


def line_plot(
    df,
    x,
    columns,
    facet,
    properties,
    fig,
    axes,
    kind="line",
    x_label=None,
    facet_is_legend=False,
    add_to_legend="",
):
    """wrapper around plot"""
    lines = list(set(df.index.get_level_values(facet)))
    lines.sort()
    default_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    for i, q in enumerate(columns):
        for j, line in enumerate(lines):
            sel = df[df.index.get_level_values(facet) == line]
            sel = idx_keep_only(sel, keep=[x])
            ax = axes[i]
            legends = properties.get("legends")
            if facet_is_legend == "final":
                legend = add_to_legend
            if facet_is_legend == "add":
                legend = line + "_" + add_to_legend
            if not add_to_legend:
                legend = line
            f = sel.sort_index().plot(
                legend=True,
                label=legend,
                lw=3,
                ms=10,
                kind=kind,
                ax=ax,
                linestyle=properties.get("linestyle", lambda x: "-")(line),
                marker=properties["marker"](line),
                color=properties.get("color", lambda x: default_colors[j])(line),
            )

            if x_label:
                ax.set_xlabel(x_label[i])
            if legends:
                plt.legend(legends)
            plt.legend(handlelength=5)


def write_figure_readme(path, fn, args):
    md_fn = fn.replace("png", "md")
    with open(path / md_fn, "w") as md_fh:
        md_fh.write("# {}\n".format(args["case"]))
        md_fh.write("![figure]({})\n".format(fn))
        md_fh.write("| machine name | {} |\n".format(args["nodes"]))
        md_fh.write("| case | {} |\n".format(args["case"]))
        md_fh.write("| ranks | {} |\n".format(args["ranks"]))


def ogl_plot(
    df,
    x,
    ys,
    x_label,
    idxs,
    facets,
    host,
    case,
    path,
    ylog=False,
    xlog=False,
    speedup_base=None,
    y_labels=[],
    drop_base=True,
    facet_is_legend="final",
    add_to_fn="",
    revision="",
    campaign="",
):
    drop_indices = [
        "node",
        "mpiRank_of",
        "processes",
        "omp_threads",
        "mpiRank_gko",
        "executor_p",
        "mpi_ranks",
        "dofs_per_rank",
    ]
    ignore_indices = []
    if speedup_base:
        for idx, val in speedup_base:
            if idx in drop_indices:
                drop_indices.remove(idx)
    if x in drop_indices:
        drop_indices.remove(x)
    if facets[0] in drop_indices:
        drop_indices.remove(facets[0])
        ignore_indices.append(facets[0])
    df = df.drop(["log_id", "timestamp", "case"], axis=1)
    y_labels = y_labels if y_labels else ys
    facet = facets[0]
    other_facet = facets[1] if len(facets) == 2 else ""
    if other_facet:
        lines = set(df.index.get_level_values(other_facet))
        dfs = [idx_query(df, [(facets[1], val, True)]) for val in lines]
    else:
        dfs = [df]
        lines = [False]

    for y, y_label in zip(ys, y_labels):
        columns = [y]
        fig, axes = plt.subplots(
            1,
            len(columns),
            figsize=(7 * len(columns), 7),
            sharex=False,
            sharey=False,
            gridspec_kw={"wspace": 0.5},
            subplot_kw={"frameon": True},
        )
        axes = [axes]
        for (df_, add_to_legend, ls) in zip(dfs, lines, ["-", "--", "-."]):
            add_to_legend = str(add_to_legend) if add_to_legend else False

            df_filt = idx_query(df_, idxs)
            # get properties for props file
            nodes = set(df_filt.index.get_level_values("node"))
            mpi_ranks = set(df_filt.index.get_level_values("mpi_ranks"))
            args = {"nodes": nodes, "case": case, "ranks": mpi_ranks}

            if speedup_base:
                sel = compute_speedup(
                    df_filt,
                    speedup_base,
                    drop_indices=drop_indices,
                    ignore_indices=ignore_indices,
                ).sort_index()
                if drop_base:
                    sel = idx_query(
                        sel, [(speedup_base[0][0], speedup_base[0][1], False)]
                    )
            else:
                sel = df_filt.sort_index()

            line_plot(
                sel[y],
                x=x,
                columns=columns,
                facet=facet,
                fig=fig,
                axes=axes,
                facet_is_legend=facet_is_legend,
                add_to_legend=add_to_legend,
                properties={
                    # "legends": ["2", "3", "4"],
                    "linestyle": lambda x: ls,
                    "marker": lambda x: "P",
                    "title": lambda x: ["Linear Solve P"][x],
                },
            )

        for ax in axes:
            ax.grid(True, axis="x", which="both", alpha=0.2)
            ax.grid(True, axis="y", which="both", alpha=0.2)
            ax.grid(True, axis="y", which="minor", alpha=0.2)
            ax.set_xlabel(x_label)
        y_label = "Speed up '{}' [-]".format(y) if speedup_base else y_label
        axes[0].set_ylabel(y_label)
        if ylog:
            axes[0].set_yscale("log")
        if xlog:
            axes[0].set_xscale("log")

        y = (
            y_label.replace(" ", "_")
            .replace("[-]", "")
            .replace("'", "")
            .replace("/", "_by_")
        )
        if speedup_base:
            y += "_over_" + str(speedup_base[0][1])
        fn = (
            host
            + "_"
            + campaign
            + "_"
            + y
            + "_by_"
            + facet
            + "_rev_"
            + other_facet
            + "_"
            + revision
            + "_"
            + add_to_fn
            + ".png"
        )
        fn = fn.replace("__", "_")
        print("save ", fn)
        write_figure_readme(path, fn, args)
        plt.savefig(path / fn, bbox_inches="tight")
