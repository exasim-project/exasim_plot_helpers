#!/usr/bin/env python3


from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import warnings

from obr.core.queries import query_to_dict


@dataclass
class JobGroup:
    """Combines queries and  plot properties for the corresponding
    query results
    """

    name: str
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


def group_jobs(jobs: list, plot_groups: list[JobGroup]) -> list[JobGroup]:
    """groups a list of jobs into several groups"""
    for group in plot_groups:
        grouped_jobs = []
        for queries in group.queries:
            grouped_jobs += query_to_dict(jobs, queries, strict=True)
        grouped_job_ids = [j.id for j in grouped_jobs]
        group.jobs += [job for job in jobs if job.id in grouped_job_ids]
    return plot_groups


def idx_larger_query(df, idx, val):
    """Shortcut to select specific index."""
    return df[df.index.get_level_values(idx) > val]


class equal:
    def __call__(self, a, b):
        return a == b

    def __repr__(self):
        return "=="


class not_equal:
    def __call__(self, a, b):
        return a != b

    def __repr__(self):
        return "!="


@dataclass
class DFQuery:
    """A query for dataframe indices"""

    idx: str
    val: Any
    op: Any = equal()

    def __repr__(self):
        try:
            op = self.op.__repr__()
        except:
            op = self.op.__name__
        return f"{self.idx}{op}{self.val}"


def idx_query_mask(df: pd.DataFrame, queries: list[DFQuery]) -> npt.NDArray:
    """perform idx queries but returns just the mask"""
    mask = np.full(len(df), True)
    for q in queries:
        mask = mask & idx_query_single_mask(df, q)
    return mask


def idx_query_single_mask(df: pd.DataFrame, q: DFQuery):
    """Shortcut to select specific index."""
    return q.op(a=df.index.get_level_values(q.idx), b=q.val)


def idx_query(df: pd.DataFrame, queries: list[DFQuery]):
    """Shortcut to query rows with specified value in index"""
    return df[idx_query_mask(df, queries)]


def calc_nodes(df, sel, masks):
    """add new index named nodes

    sel: select cases by this index
    masks: maps from index value to ranks per node ie
            [[CUDA, 4], [Default, 76]]
    """
    df["nodes"] = df.index.get_level_values("mpi_ranks")
    for sel_value, value in masks:
        mask = df.index.get_level_values(sel) == sel_value
        df.loc[mask, "nodes"] = df.loc[mask, "nodes"] / value
    df = df.set_index("nodes", append=True)
    return df


def merge_index(df, first, second, name_map):
    """takes two index columns and replaces values according to map"""
    e = df.index.get_level_values(second)
    b = df.index.get_level_values(first)
    df.index = df.index.droplevel(first)

    merged = [name_map[str(a) + str(b)] for a, b in zip(e, b)]
    df[first] = merged
    return df.set_index(first, append=True)


def idx_keep_only(df: pd.DataFrame, keep: list[str]) -> pd.DataFrame:
    """drops all index columns except the columns specified in keep"""
    drop_idxs = [x for x in df.index.names if x not in keep]
    return df.reset_index(level=drop_idxs, drop=True)


def compute_full_node_normalize(df, ref: list[DFQuery]):
    """Compute the speedup compared to the full node execution on same host"""

    def get_reference_value(x):
        """This function is used within apply, hence host are already the same"""
        full_node_ranks = max(
            idx_query(x, ref).index.get_level_values("numberOfSubDomains")
        )
        ranks_query = [DFQuery("numberOfSubDomains", full_node_ranks)]
        return compute_speedup(x, ref + ranks_query)

    return df.groupby(["host"]).apply(get_reference_value)


def compute_speedup(
    df, ref: list[DFQuery], drop_indices=None, ignore_indices=None, inverse=False
):
    """Compute and return the speedup compared to a reference.

    Parameters:
        df:

    """
    df = deepcopy(df)
    if df.empty:
        raise ValueError("cannot compute speedup on empty dataframes")

    if drop_indices:
        for idx in drop_indices:
            if idx not in df.index.names:
                continue
            df.index = df.index.droplevel(idx)

    reference = idx_query(df, ref)
    if not reference.index.is_unique:
        warnings.warn("Reference should have a unique idx")

    ref_drop_idxs = [x.idx for x in ref]
    reference.index = reference.index.droplevel(ref_drop_idxs)

    if ignore_indices:
        reference.index = reference.index.droplevel(ignore_indices[0])

    def dropped_divide(df):
        df = deepcopy(df)
        df.index = df.index.droplevel(ref_drop_idxs)
        return df

    def apply_func(x):
        if ignore_indices:
            ignored_idx = x.index.get_level_values(ignore_indices[0])
            x.index = x.index.droplevel(ignore_indices[0])
        try:
            # make sure that the number of rows is correct
            divisor = dropped_divide(x)
            if inverse:
                ret = np.divide(divisor, reference, where=divisor.dtypes.eq(np.isreal))
            else:
                divisor_non_num = divisor.select_dtypes(exclude=np.number)
                ref = reference.select_dtypes(include=np.number)
                divisor = divisor.select_dtypes(include=np.number)
                ret = ref / divisor
                try:
                    for col in divisor_non_num.columns:
                        ret[col] = divisor_non_num[col]
                except:
                    print(col)
                    pass
        except Exception as e:
            print(e)
            print(f"division failed:\nref = {reference}\ndiv = {divisor}")
        if ignore_indices:
            ret[ignore_indices[0]] = ignored_idx.values
            ret.set_index(ignore_indices[0], append=True, inplace=True)
        return ret

    res = df.groupby(level=ref_drop_idxs).apply(apply_func)

    return res
