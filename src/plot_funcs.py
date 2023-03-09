import matplotlib.pyplot as plt
from helpers import idx_query, idx_keep_only, compute_speedup


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
