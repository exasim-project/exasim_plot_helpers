#!/usr/bin/env python3

import numpy as np
import pandas as pd
from dataclasses import dataclass


def idx_larger_query(df, idx, val):
    """Shortcut to select specific index."""
    return df[df.index.get_level_values(idx) > val]


def equal(a, b):
    return a == b


def not_equal(a, b):
    return a != b


@dataclass
class query:
    idx
    val
    operator = equal


def idx_query_mask(df: pd.DataFrame, queries: list[query]) -> np.array[bool]:
    """perform idx queries but returns just the mask"""
    mask = np.full(len(df), True)
    for q in queries:
        idx, val = keys
        mask = mask & idx_query_single_mask(df, idx, val)
    return mask


def idx_query_single_mask(df: pd.DataFrame, q: query):
    """Shortcut to select specific index."""
    return q.op(df.index.get_level_values(q.idx), q.val)


def idx_query(df: pd.DataFrame, queries: list[query]):
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


def compute_speedup(df, ref: list[queries], drop_indices=None, ignore_indices=None):
    """Compute and return the speedup compared to a reference."""
    from copy import deepcopy

    df = deepcopy(df)

    if drop_indices:
        for idx in drop_indices:
            if idx not in df.index.names:
                continue
            df.index = df.index.droplevel(idx)

    reference = idx_query(df, ref)
    ref_drop_idxs = [x[0] for x in ref]
    reference.index = reference.index.droplevel(ref_drop_idxs)
    if ignore_indices:
        reference.index = reference.index.droplevel(ignore_indices[0])

    def dropped_divide(df):
        from copy import deepcopy

        df = deepcopy(df)
        df.index = df.index.droplevel(ref_drop_idxs)
        return df

    def apply_func(x):
        if ignore_indices:
            ignored_idx = x.index.get_level_values(ignore_indices[0])
            x.index = x.index.droplevel(ignore_indices[0])

        ret = reference / dropped_divide(x)
        if ignore_indices:
            ret[ignore_indices[0]] = ignored_idx.values
            ret.set_index(ignore_indices[0], append=True, inplace=True)
        return ret

    res = df.groupby(level=ref_drop_idxs).apply(apply_func)

    return res
