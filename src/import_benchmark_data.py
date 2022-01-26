#!/usr/bin/env python3

import os
import pandas as pd
import Owls as ow

from pathlib import Path
from packaging import version

from helpers import idx_query


def clean_hash(s):
    return s.replace("hash: ", "").replace("\n", "")


def read_logs(folder):
    """Reads the logs file in the folder."""

    log_handles = []
    root, _, logfiles = next(os.walk(folder))
    root = Path(root)

    end_of_log_marker = "=" * 80 + "\n"
    log_content = []
    for logfile in logfiles:
        with open(root / logfile) as log_handle:
            log_content += log_handle.read().split(end_of_log_marker)
    return log_content


def parse_log_strings(log_content):
    """Convert a list of of log strings to a dictionary of foam frames."""

    logs = {}
    for log in log_content:
        log_ = log.split("\n")
        log_header = log_[0]
        log_body = log_[1:]
        logs.update({clean_hash(log_header): log_body})

    keys = {"linear solve " + f: ["linear_solve_" + f] for f in ["U", "p"]}
    keys.update(
        {
            "ExecutionTime ": ["ExecutionTime", "ClockTime"],
            "init_precond": ["Precond_Proc", "init_precond"],
            "update_host_matrix ": ["Update_Proc", "update_host_matrix"],
            "]solve ": ["Solve_Proc", "gko_solve"],
            "retrieve_results_from_device ": ["Retrieve_Proc", "retrieve_results"],
        }
    )
    keys.update(
        {
            "Solving for {}".format(f): [
                "init_residual_" + f,
                "final_residual_" + f,
                "number_iterations_" + f,
            ]
            for f in ["p", "U"]
        }
    )

    for log_hash, log_content in logs.items():
        if not log_content:
            continue
        try:
            logs[log_hash] = [ow.io.import_log_from_str(log_content, keys)]
        except Exception as e:
            print(e)
            pass

    return logs


def process_meta_data(s):
    """Given a csr file as string the function reads embeded metatdata dict."""
    if len(s) < 2:
        return None
    if not ("#" in s):
        return None
    else:
        try:
            meta_data_str = s.replace("#", "").replace("\n", "")
            metadata = eval(meta_data_str)
            if not metadata:
                return None
            if not "OBR_REPORT_VERSION" in metadata.keys():
                return None
            return metadata
        except Exception as e:
            return None


def read_ogl_data_folder(folder, min_version="0.0.0"):
    """Reads all csv files and logs from the given folder.

    returns a concatenated dataframe
    """
    # TODO refactor this
    dfs = []
    metadata = {}

    folder = Path(folder)

    logs = parse_log_strings(read_logs(folder / "Logs"))

    _, _, reports = next(os.walk(folder))
    for r in reports:
        fn = folder / r

        with open(fn) as csv_handle:
            # read metadata
            try:
                content = csv_handle.readlines()
            except:
                continue
            metadata = process_meta_data(content[1])
            if not metadata:
                continue
            obr_version = version.parse(metadata["OBR_REPORT_VERSION"])
            if obr_version < version.parse(min_version):
                continue
            metadata[fn] = metadata

        df = pd.read_csv(fn, comment="#")

        # check if reading was a success
        if len(df) == 0:
            continue
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True), metadata, logs


def import_results(
    path,
    case,
    filt=None,
    min_version="0.0.0",
    short_hostnames=False,
    reset_ogl_version=False,
):
    """Import and postprocess obr benchmark results."""
    path = Path(path)
    df, metadata, logs = read_ogl_data_folder(path / case, min_version)

    # to use pandas multiindices data and non-data columns need to be separated
    data_columns = [
        "log_id",
        "run_time",
        "setup_time",
        "number_iterations_p",
        "number_iterations_U",
        "init_linear_solve_p",
        "linear_solve_p",
        "init_linear_solve_U",
        "linear_solve_U",
    ]

    df["solver_p"] = df["solver_p"].transform(
        lambda x: x.replace("GKO", "").replace("P", "")
    )

    if short_hostnames:
        df["node"] = df["node"].transform(lambda x: short_hostnames(x))

    indices = [c for c in df.columns if c not in data_columns]

    df["linear_solve_p"] = 0
    df["linear_solve_U"] = 0

    # Refactor
    for log_hash, rets in logs.items():
        for ret in rets:
            try:
                latest_time = max(ret.index.get_level_values("Time"))
                for key in [
                    "linear_solve_p",
                    "linear_solve_U",
                    "number_iterations_p",
                    "number_iterations_U",
                    # "init_precond",
                    # "update_host_matrix",
                    # "retrieve_results",
                ]:
                    df.loc[df["log_id"] == log_hash, key] = idx_query(
                        ret, "Time", latest_time
                    )[key].mean()
                gko_keys = [
                    "init_precond",
                    "update_host_matrix",
                    "retrieve_results",
                    "gko_solve",
                ]
                for key in gko_keys:
                    if key in ret.columns:
                        df.loc[df["log_id"] == log_hash, key] = idx_query(
                            ret, "Time", latest_time
                        )[key].mean()
                    else:
                        df.loc[df["log_id"] == log_hash, key] = 0

                deltaT = ret["ClockTime"].dropna().diff().values[-1]
                df.loc[df["log_id"] == log_hash, "deltaT"] = deltaT
            except Exception as e:
                print("import_benchmark_data", e)
                pass

    # skip cases with zero run_time
    df = df[df["run_time"] > 0]

    # calculate some further metrics
    # TODO pass that as function
    df["linear_solve_p_per_iter"] = df["linear_solve_p"] / df["number_iterations_p"]
    df["linear_solve_p_percent"] = df["linear_solve_p"] / df["deltaT"] / 1e4
    df["linear_solve_p_ratio"] = df["linear_solve_p"] / df["linear_solve_U"]
    # df["cells"] = df["resolution"] ** dimension
    # df["linear_solve_p_per_cell_and_iter"] = df["linear_solve_p_per_iter"] / df["cells"]

    # reorder indices a bit
    indices[0], indices[1] = indices[1], indices[0]
    df = df.fillna(0)
    df.set_index(indices, inplace=True)

    # compute mean of all values grouped by all indices
    # this will compute mean when all indices are identical
    mean = df.groupby(level=indices).mean()

    return {"raw": df, "mean": mean, "metadata": metadata, "logs": logs}
