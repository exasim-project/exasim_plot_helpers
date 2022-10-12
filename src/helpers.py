#!/usr/bin/env python3
#
#
def idx_larger_query(df, idx, val):
    """Shortcut to select specific index."""
    return df[df.index.get_level_values(idx) > val]

def idx_query(df, queries):
    for keys in queries:
        if len(keys) == 2:
            idx, val = keys
            df = idx_query_single(df, idx, val)
        if len(keys) == 3:
            idx, val, f = keys
            if isinstance(f, bool):
                if f:
                    df = idx_query_single(df, idx, val)
                else:
                    df = idx_not_query_single(df, idx, val)                
    return df

def calc_nodes(df, sel, masks):
    """  add new index named nodes 
    
    sel: select cases by this index
    masks: maps from index value to ranks per node ie
            [[CUDA, 4], [Default, 76]]
    
    """
    df["nodes"] = df.index.get_level_values('mpi_ranks')
    for sel_value, value in masks:
        mask = df.index.get_level_values(sel) == sel_value
        df.loc[mask,"nodes"] = df.loc[mask, "nodes"]/value
    df = df.set_index("nodes", append=True)
    return df

def merge_index(df, first, second, name_map):
    """ takes two index columns and replaces values according to map """
    e = df.index.get_level_values(second)
    b = df.index.get_level_values(first)
    df.index = df.index.droplevel(first)
    
    merged = [name_map[str(a)+str(b)] for a,b in zip(e,b)]
    df[first] = merged
    return df.set_index(first,append=True)

    
def idx_query_single(df, idx, val):
    """Shortcut to select specific index."""
    return df[df.index.get_level_values(idx) == val]


def idx_not_query_single(df, idx, val):
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
            if idx not in  df.index.names:
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

