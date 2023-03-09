#!/usr/bin/env python3
import matplotlib.pyplot as plt

import os
import pandas as pd
import Owls as ow

from pathlib import Path
from packaging import version

from helpers import idx_larger_query, idx_keep_only


def clean_hash(s):
    return s.replace("hash: ", "").replace("\n", "")


def get_case_from_header(header):
    if len(header) < 15:
        return ""
    header = header[15].split(":")[-1]
    return (
        header.replace(
            "/hkfs/home/project/hk-project-test-fine/eq4036/data/code/polimi-nose-labbook/nose/preconditioner/Variation_matrix_solver",
            "",
        )
        .replace("Variation_", "")
        .replace("base", "")
        .replace("_gko", "")
        .replace("_of", "")
    )


def get_slurm_reports(s):
    """get everything from 25*= JOB FEEDBACK to hash"""
    slurm_start = 0
    slurm_end = 0

    for i, line in enumerate(s):
        if "=" * 10 in line and "JOB FEEDBACK" in line:
            slurm_start = i
        if "hash: " in line:
            slurm_end = i
            break

    return s[slurm_end], s[slurm_start : slurm_end - 1], s[slurm_end + 1 :]


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
    case = ""
    for log in log_content:
        log_ = log.split("\n")
        log_header, slurm_report, log_body = get_slurm_reports(log_)
        case = get_case_from_header(log_body)
        try:
            end = log_body[-2]
        except:
            end = ""
        if not "Finalising" in end:
            continue
        logs.update({clean_hash(log_header): (log_body, slurm_report, case)})

    keys = {"linear solve " + f: ["linear_solve_" + f] for f in ["U", "p"]}
    keys.update(
        {
            "ExecutionTime ": ["ExecutionTime", "ClockTime"],
            "init_precond": ["Precond_Proc", "init_precond"],
            "update_host_matrix_data": ["Update_Proc", "update_host_matrix"],
            "]solve": ["Solve_Proc", "gko_solve"],
            "copy_x_back": ["Retrieve_Proc", "retrieve_results"],
            "delta t build": ["build_dist", "write_mat_data", "build_repart", "gather"],
            "piso setup": ["poisson_assembly"],
            "momentum setup": ["momentum_assembly"],
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
    invalid = []
    for log_hash, (log_content, slurm_report, case) in logs.items():
        if not log_content:
            continue
        try:
            logs[log_hash] = [
                ow.io.import_log_from_str(log_content, keys),
                slurm_report,
                case,
            ]
        except Exception as e:
            print("parse_log_strings", e)
            invalid.append(log_hash)
            pass

    for i in invalid:
        print("popping", i)
        logs.pop(i)
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


def read_ogl_data_folder(folder, filt, min_version="0.0.0"):
    """Reads all csv files and logs from the given folder.

    returns a concatenated dataframe
    """
    # TODO refactor this
    dfs = []
    metadata = {}

    folder = Path(folder)
    logs_folder = folder / "Logs"
    if not logs_folder.exists():
        print(folder, "does not contain a Logs folder")
        raise Exeception

    print("importing Logs", folder / "Logs")

    logs = parse_log_strings(read_logs(folder / "Logs"))

    _, _, reports = next(os.walk(folder))
    for r in reports:
        fn = folder / r
        if filt in r:
            continue

        with open(fn) as csv_handle:
            # read metadata
            try:
                content = csv_handle.readlines()
            except Exception as e:
                print(e)
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


def post_process_df(df):
    df["linear_solve_p_per_iter"] = df["linear_solve_p"] / df["number_iterations_p"]
    df["linear_solve_p_percent"] = df["linear_solve_p"] / df["deltaT"] / 1e4
    df["linear_solve_p_ratio"] = df["linear_solve_p"] / df["linear_solve_U"]
    df["cells"] = df["resolution"] ** 3
    df["linear_solve_p_per_cell_and_iter"] = df["linear_solve_p_per_iter"] / df["cells"]

    df["linear_solve_U_per_iter"] = df["linear_solve_U"] / df["number_iterations_U"]
    df["linear_solve_U_per_cell_and_iter"] = df["linear_solve_U_per_iter"] / df["cells"]

    df["gko_solve_p_per_iter"] = df["gko_solve"] / df["number_iterations_p"]
    df["gko_solve_p_per_cell_and_iter"] = df["gko_solve_p_per_iter"] / df["cells"]

    df["gko_overhead"] = df["linear_solve_p"] - df["gko_solve"]
    df["gko_overhead_percent"] = df["gko_overhead"] / (
        df["gko_overhead"] + df["gko_solve"]
    )
    df["gko_overhead_per_iter"] = df["gko_overhead"] / df["linear_solve_p_per_iter"]
    df["gko_overhead_per_cell"] = df["gko_overhead"] / (df["cells"])
    df["gko_overhead_per_cell_and_iter"] = df["gko_overhead"] / (
        df["cells"] * df["linear_solve_p_per_iter"]
    )
    df["dofs_per_rank"] = df["cells"] / df["mpi_ranks"]
    df["gko_solve_p_per_local_dof_and_iter"] = (
        df["linear_solve_p_per_iter"] / df["dofs_per_rank"]
    )

    return df


def import_results(
    path,
    case,
    campaign,
    revision,
    filt=None,
    min_version="0.0.0",
    short_hostname_map=False,
    reset_ogl_version=False,
    transform_resolution=True,
    resolution_map=None,
    skip_zero_runtime=False,
):
    """Import and postprocess obr benchmark results."""
    path = Path(path) / revision / case / campaign
    if not (path).exists():
        print(path, "not existent")
        raise Exception
    df, metadata, logs = read_ogl_data_folder(path, filt, min_version)

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
        "timestamp",
        "update_host_matrix",
        "retrieve_results",
        "gather",
        "poisson_assembly",
        "momentum_assembly",
    ]

    df["solver_p"] = df["solver_p"].transform(
        lambda x: x.replace("GKO", "").replace("P", "")
    )

    def short_hostname(x):
        for key, value in short_hostname_map.items():
            if key in x:
                return value

    if short_hostname_map:
        df["node"] = df["node"].transform(lambda x: short_hostname(x))

    if transform_resolution:
        df["resolution"] = df["resolution"].transform(lambda x: x**3)

    if resolution_map:
        df["resolution"] = df["resolution"].transform(lambda x: resolution_map[x])

    indices = [c for c in df.columns if c not in data_columns]
    # add derived indices
    indices += ["cells", "dofs_per_rank"]

    df["linear_solve_p"] = 0
    df["linear_solve_U"] = 0
    df["EnergyConsumed"] = 0

    # Refactor
    for log_hash, rets in logs.items():
        rets, slurm_logs, case = rets[0:-2], rets[-2], rets[-1]
        for ret in rets:
            try:
                # first_time = max(ret.index.get_level_values("Time"))
                # first_time = min(ret.index.get_level_values("Time"))
                first_time = ret.index.get_level_values("Time")[1]
                last_time = ret.index.get_level_values("Time")[-1]

                for key in [
                    "linear_solve_p",
                    "linear_solve_U",
                    "number_iterations_p",
                    "number_iterations_U",
                    # "init_precond",
                    # "update_host_matrix",
                    # "retrieve_results",
                ]:
                    df.loc[df["log_id"] == log_hash, key] = idx_larger_query(
                        ret, "Time", first_time
                    )[key].mean()
                gko_keys = [
                    "init_precond",
                    "update_host_matrix",
                    "retrieve_results",
                    "gko_solve",
                    "poisson_assembly",
                    "gather",
                    "momentum_assembly",
                ]
                df.loc[df["log_id"] == log_hash, "last_time"] = last_time
                for key in gko_keys:
                    if key in ret.columns:
                        df.loc[df["log_id"] == log_hash, key] = idx_larger_query(
                            ret, "Time", first_time
                        )[key].mean()
                    else:
                        df.loc[df["log_id"] == log_hash, key] = 0

                deltaT = ret["ClockTime"].dropna().diff().values[2:].sum()
                df.loc[df["log_id"] == log_hash, "deltaT"] = deltaT
                df.loc[df["log_id"] == log_hash, "case"] = case
                for line in slurm_logs:
                    if "Energy Consumed" in line:
                        tokens = line.split(" ")
                        df.loc[df["log_id"] == log_hash, "EnergyConsumed"] = float(
                            tokens[-2]
                        )

            except Exception as e:
                print("import_benchmark_data", e)
                pass

    # skip cases with zero run_time
    if skip_zero_runtime:
        df = df[df["run_time"] > 0]
    # calculate some further metrics
    # TODO pass that as function
    df = post_process_df(df)

    # reorder indices a bit
    indices[0], indices[1] = indices[1], indices[0]
    df = df.fillna(0)
    df.set_index(indices, inplace=True)

    # compute mean of all values grouped by all indices
    # this will compute mean when all indices are identical
    mean = df.groupby(level=indices).mean()

    return {"raw": df, "mean": mean, "metadata": metadata, "logs": logs}
