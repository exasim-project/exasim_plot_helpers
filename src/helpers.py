#!/usr/bin/env python3
#
#
def idx_query(df, idx, val):
    """Shortcut to select specific index."""
    return df[df.index.get_level_values(idx) == val]


def idx_not_query(df, idx, val):
    """Shortcut to filter specific index."""
    return df[df.index.get_level_values(idx) != val]


def idx_keep_only(df, keep):
    drop_idxs = [x for x in df.index.names if x not in keep]
    return df.reset_index(level=drop_idxs, drop=True)


def compute_speedup(df, ref, drop_indices=None, ignore_indices=None):
    """Compute and return the speedup compared to a reference."""
    from copy import deepcopy

    df = deepcopy(df)
    if drop_indices:
        for idx in drop_indices:
            df.index = df.index.droplevel(idx)

    reference = idx_query(df, ref[0], ref[1])
    reference.index = reference.index.droplevel([ref[0]])
    reference.index = reference.index.droplevel(ignore_indices[0])

    def dropped_divide(df):
        from copy import deepcopy

        df = deepcopy(df)
        df.index = df.index.droplevel(ref[0])
        return df

    def apply_func(x):
        ignored_idx = x.index.get_level_values(ignore_indices[0])
        x.index = x.index.droplevel(ignore_indices[0])

        ret = reference / dropped_divide(x)
        ret[ignore_indices[0]] = ignored_idx.values

        ret.set_index(ignore_indices[0], append=True, inplace=True)
        return ret

    res = df.groupby(level=ref[0]).apply(apply_func)

    return res
