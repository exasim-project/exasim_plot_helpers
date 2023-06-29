import os
import traceback
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from helpers import (DFQuery, compute_speedup, idx_keep_only, idx_query,
                     idx_query_mask)


@dataclass
class PlotCampaignSpecifications:
    """Our measurements are usually indexed by nCells, solver, executor, nSubDomains, preconditioner
    measurement campaign might select jobs that are distinguished by other categories which are not part of the index. This means the individual results need to be distinguishable a datacolumn eg by the jobid, ogl version hash and another index.

    Examples:
        - when updating OGL the version hash is different, after runnning the same jobs all indices
        and jobids are still the same
        - when comparing different simulation cases, again the indices might be same
        - exploring a new feature of OGL will create different jobs but the indices are constant

    This dataclass holds the corresponding dataframes for each measurement
    """

    campaign_name: str
    plot_collection: dict  # measurement_name: list[PlotSpecifications]
    dfs: dict[Any]  # measurement_name :  df
    overwrite_properties: dict  # base: {'linestyle':'--'}

    @classmethod
    def generate(cls):
        pass


@dataclass
class PlotSpecification:
    """Specifies what to plot from a dataframe via DFQueries and basic plot properties such as linestyle and legend"""

    legend: str
    queries: list = field(default_factory=list)
    plot_properties: dict = field(default_factory=dict)
    df: Any = None
    color_cycle: Any = field(
        default_factory=lambda: [
            "tab:blue",
            "tab:orange",
            "tab:green",
            "tab:red",
            "tab:purple",
        ]
    )


@dataclass
class FacetedPlotSpecification:
    """Collects a list of PlotSpecification for faceting a plot over a dataframe idx"""

    plotGroups: list[PlotSpecification]
    facet_idx: str

    @classmethod
    def generate(cls, fix_queries, fix_properties, facet_idx, variable_properties):
        groups = []
        variable_queries = []
        for k, v in variable_properties.items():
            legend = v.pop("legend")
            variable_queries.append((legend, [DFQuery(idx=facet_idx, val=k)], v))
        for legend, var_query, var_properties in variable_queries:
            properties = fix_properties
            properties.update(var_properties)
            groups.append(
                PlotSpecification(
                    legend=legend,
                    queries=fix_queries + var_query,
                    plot_properties=deepcopy(properties),
                )
            )
        return FacetedPlotSpecification(plotGroups=groups, facet_idx=facet_idx)

    def get_facet_values(self, df):
        facet_values = []
        for plot_group in self.plotGroups:
            query_mask = idx_query_mask(df, plot_group.queries)
            filtered_df = df[query_mask]
            facet_value = set(filtered_df.index.get_level_values(self.facet_idx))

            if not len(facet_value) == 1:
                raise ValueError(
                    f"{self.facet_idx} does not produce a unique set of facet_values  {facet_values}"
                )
            facet_values.append(list(facet_value)[0])
        return facet_values


class PlotDispatcher:
    def __init__(self, repo_path, storage_schema="image.png"):
        self.repo_path = repo_path
        self.storage_schema = storage_schema

    # f"{self.repo_path}/{case}/figs/{system_name}/{func.__name__}_{func_args_str}_{append_to_fn}.png"

    def dispatch_plot(
        self, func, campaign, ax_handler=lambda x: x, append_to_fn="", **kwargs
    ):
        """Trys to generate a plot and writes to file based on func.__name__ and args to a file.
        It iterates all plot_collections and corresponding dataframes
        """
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 5), sharey=True)
        labels = []
        try:
            for measurement, df in campaign.dfs.items():
                plot_properties = campaign.plot_collection[measurement]
                print(kwargs)
                fig, axes, labels = func(
                    df,
                    plot_properties,
                    kwargs.pop("x"),
                    kwargs.pop("y"),
                    fig_axes=(fig, axes),
                    **kwargs,
                )
            func_args_str = "_".join([f"{k}={v}" for k, v in kwargs.items()])
            ax_handler(axes)
            fn = self.storage_schema
            print(f"Save {fn}")
            fig.savefig(
                Path(fn),
                bbox_inches="tight",
            )
            return fig, axes
        except Exception as e:
            print("failed to plot", __name__, func.__name__, e)
            traceback.print_tb(e.__traceback__)

            return fig, axes


def facets_over_x(
    df: pd.DataFrame, grouper: FacetedPlotSpecification, x: str, y: str, fig_axes=None
):
    """Plot faceted by a set of plot groups

    Parameters:
     - legend: a string formatable by query dict key
     - queries: a list of PlotSpecifications
     - x: name of the index to plot over
     - y: name of the column to plot

    Returns:
     - the figure
     - the axes
    """

    if not fig_axes:
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 5), sharey=True)
    else:
        fig, axes = fig_axes

    labels = []

    for plot_group in grouper.plotGroups:
        query_mask = idx_query_mask(df, plot_group.queries)
        filtered_df = df[query_mask]

        # keep only x as indices to keep plot axis clean
        filtered_df = idx_keep_only(filtered_df, [x])

        filtered_df[y].plot(ax=axes, **plot_group.plot_properties)
        labels.append(plot_group.legend)

    axes.legend(labels)

    return fig, axes, labels


def facets_relative_to_base_over_x(
    df: pd.DataFrame,
    grouper: FacetedPlotSpecification,
    x: str,
    y: str,
    fig_axes=None,
    base_query: PlotSpecification = None,
):
    """faceted plot normalised values over x

    Parameters:
     - queries: a dictionary of query name and a list queries
     - x: name of the index to plot over
     - y: name of the column to plot

    Returns:
     - the figure
     - the axes
     - the labels
    """
    # generate over different partitionings
    # get base_ranks TODO find a generic way to do this
    if not fig_axes:
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 5), sharey=True)
    else:
        fig, axes = fig_axes

    base_query_mask = idx_query_mask(df, base_query.queries)
    not_base_query_mask = np.logical_not(base_query_mask)

    # get available facets for non base case
    facet_values = grouper.get_facet_values(df[not_base_query_mask])

    labels = []
    for plot_group, facet_value in zip(grouper.plotGroups, facet_values):
        # compute individual speed up
        # pre filter DataFrame to contain either base or facet values

        average_df = False
        if df[base_query_mask].index.has_duplicates:

            warnings.warn(
                f"Found non unique indices for base query DataFrame {df[base_query_mask].index}"
            )
            average_df = True

        if df[idx_query_mask(df, plot_group.queries)].index.has_duplicates:

            warnings.warn(
                f"Found non unique indices for faceted query DataFrame {df[idx_query_mask(df, plot_group.queries)].index}"
            )

            average_df = True

        filtered_df = df[idx_query_mask(df, plot_group.queries) | base_query_mask]

        if average_df:
            index_names = filtered_df.index.names
            filtered_df = filtered_df.groupby(filtered_df.index).mean()
            index_tuples = filtered_df.reset_index()["index"]
            filtered_df.index = pd.MultiIndex.from_tuples(
                index_tuples, names=list(index_names)
            )

        speedup = compute_speedup(
            filtered_df,
            base_query.queries,
            # ignore_indices=[facet],
            drop_indices=["solver"],
        )

        # remove reference values
        speedup = speedup[np.logical_not(idx_query_mask(speedup, base_query.queries))]

        # keep only x as indices to keep plot axis clean
        speedup = idx_keep_only(speedup, [x])

        speedup[y].plot(ax=axes)  # , **plot_properties)
        labels.append(plot_group.legend.format(value=facet_value))

    axes.legend(labels)
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
