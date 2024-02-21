#!/usr/bin/env python3


import logging
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
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


class geq:
    def __call__(self, a, b):
        return a >= b

    def __repr__(self):
        return ">="


class lt:
    def __call__(self, a, b):
        return a < b

    def __repr__(self):
        return "<"


def val_queries(df, idx_val_pairs: list[tuple]):
    """Performs queries on a dataframe based on column values instead of indices
    Parameter:
     - df the dataframe for which to compute the mask
     - idx_val_pairs a list of tuples of (idx, value, predicate)
    Returns: The resulting DataFrame
    """
    return df[val_queries_mask(df, idx_val_pairs)]


def val_queries_mask(df, idx_val_pairs: list[tuple]):
    """Performs queries on a dataframe based on column values instead of indices
    Parameter:
     - df the dataframe for which to compute the mask
     - idx_val_pairs a list of tuples of (idx, value, predicate)
    Returns the resulting mask
    """
    mask = np.full(df.shape[0], True)
    for idx, val, pred in idx_val_pairs:
        mask = np.logical_and(mask, pred(df[idx], val))
    return mask


@dataclass
class DFQuery:
    """A query for dataframe indices"""

    idx: str
    val: Any
    op: Any = equal()

    def to_tuple():
        return (idx, val, op)

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
    df,
    refs: list[dict[DFQuery]],
    drop_indices=None,
    ignore_indices=None,
    inverse=False,
    exclude=None,
) -> pd.DataFrame:
    """Compute and return the speedup compared to a reference.

    Parameters:
        df: the dataframe to compute the speedup on
        exclude: List of columns to leave intact
    """
    # Some debug stuff
    if False:
        logging.warning(f"input dataframe {df.to_dict()}")
        logging.warning(f"input dataframe {df.index.names}")
        logging.warning(f"input refs {refs}")
        logging.warning(f"input ignore_indices {ignore_indices}")
        logging.warning(f"input exclude {exclude}")

    # TODO
    # df = pd.DataFrame({'Max Speed': [390., 350., 30., 20.], 'Ref Speed': [2., 2., 4., 4.], "Host": ["a", "c", "b", "d"], "nCells":[1,2,3,4], "foo": [0,1,0,1]})
    # base = [
    #      {
    #         "case": [
    #              helpers.DFQuery(idx="Host", val="a"),
    #          ],
    #          "base":[ helpers.DFQuery(idx="Host", val="b")]
    #      },
    #      {
    #         "case": [helpers.DFQuery(idx="Host", val="c")],
    #         "base": [helpers.DFQuery(idx="Host", val="d"), helpers.DFQuery(idx="nCells", val=4)]
    #      },
    #      ]
    # produces nans
    #                        Max Speed  Ref Speed
    #   Host nCells foo
    #   a    1      0          NaN        NaN <- why are the results nan
    #        3      0          NaN        NaN <- this is a wrong row
    #   c    2      1     0.057143        2.0

    # TODO check if ref idx are actually indices of the dataframe

    excluded = None
    do_exclude = False

    if exclude and all([c in df.columns for c in exclude]):
        do_exclude = True
        excluded = df[exclude]

    def dropped_divide(df):
        df = deepcopy(df)
        df.index = df.index.droplevel(ref_drop_idxs)
        return df

    class ApplyFunc:
        def __init__(self, reference):
            self.reference = reference

        def get_apply_func(self):
            def apply_func(x):
                if ignore_indices:
                    ignored_idx = x.index.get_level_values(ignore_indices[0])
                    x.index = x.index.droplevel(ignore_indices[0])
                try:
                    # make sure that the number of rows is correct
                    divisor = dropped_divide(x)
                    if inverse:
                        ret = np.divide(
                            divisor, self.reference, where=divisor.dtypes.eq(np.isreal)
                        )
                    else:
                        divisor_non_num = divisor.select_dtypes(exclude=np.number)
                        ref = self.reference.select_dtypes(include=np.number)
                        divisor = divisor.select_dtypes(include=np.number)
                        ret = ref / divisor
                        try:
                            for col in divisor_non_num.columns:
                                ret[col] = divisor_non_num[col]
                        except:
                            print("exeception in col", col)
                            pass
                except Exception as e:
                    print(e)
                    print(f"division failed:\nref = {reference}\ndiv = {divisor}")
                if ignore_indices:
                    ret[ignore_indices[0]] = ignored_idx.values
                    ret.set_index(ignore_indices[0], append=True, inplace=True)
                return ret

            return apply_func

    df = deepcopy(df)

    if df.empty:
        raise ValueError("cannot compute speedup on empty dataframes")

    if drop_indices:
        for idx in drop_indices:
            if idx not in df.index.names:
                continue
            df.index = df.index.droplevel(idx)

    res = pd.DataFrame()
    for records in refs:
        ref = records["base"]
        case = records["case"]

        reference = deepcopy(idx_query(df, ref))
        if reference.empty:
            logging.warning(
                f"Reference DataFrame for query {ref} is empty skipping, skipping"
            )
            continue

        if not reference.index.is_unique:
            warning = """Potential problem in dataframe detected:
            Reference should have a unique idx!

            Reference: {}
            Reference query {}

            """.format(
                reference, ref
            )
            logging.warning(warning)

        ref_drop_idxs = [x.idx for x in ref]
        reference.index = reference.index.droplevel(ref_drop_idxs)

        if ignore_indices:
            reference.index = reference.index.droplevel(ignore_indices[0])

        res = pd.concat(
            [
                res,
                idx_query(df, case)
                .groupby(level=ref_drop_idxs)
                .apply(ApplyFunc(reference).get_apply_func()),
            ]
        )

    if res.empty:
        logging.warning(f"Resulting DataFrame is empty")
        logging.warning(f"Initial Dataframe {df}\n refs {refs}")

    if do_exclude:
        for col in excluded.columns:
            res[col] = excluded[col]

    return res
