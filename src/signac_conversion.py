import os
import flow
import pandas as pd

from obr.core.queries import Query, query_to_dataframe
from obr.signac_wrapper.operations import OpenFOAMProject
from Owls.parser.LogFile import LogKey


def build_default_queries() -> list:
    l = list(
        map(
            lambda x: Query(key=x),
            [
                "solver",
                "host",
                "campaign",
                "tags",
                "timestamp",
                "preconditioner",
                "executor",
                "SolveP",
                "nCells",
                "numberOfSubDomains",
            ],
        )
    )
    return l



def build_OGLAnnotationKeys(fields: list[str]) -> list[str]:
    """Function to generate search keys for log files based on field name"""
    return [
        key.format(field)
        for key in [
            "{}: update_local_matrix_data:",
            "{}: update_non_local_matrix_data:",
            "{}_matrix: call_update:",
            "{}_rhs: call_update:",
            "{}: init_precond:",
            "{}: generate_solver:",
            "{}: solve:",
            "{}: copy_x_back:",
            "{}: solve_multi_gpu",
        ]
        for field in fields
    ]


def build_transport_eqn_keys() -> list[LogKey]:
    # columns names for generated DataFrame
    col_iter = ["Initial residual", "Final residual", "Number iterations"]

    # post fix for pressure eqns
    p_steps = [" p", " pFinal"]

    # post fix for momentum components
    U_components = [" Ux", " Uy", " Uz"]

    pIter = LogKey("Solving for p", columns=col_iter, post_fix=p_steps)
    UIter = LogKey("Solving for U", columns=col_iter, post_fix=U_components)
    return [pIter, UIter]


def generate_log_keys() -> dict:
    """This function generates various LogKey instances to analyze log files. Here several types
    of LogKeys are considered:
        1. transp_eqn_keys: for log entries of the form Solving for ?: init, final res. iter
        2. annotation_keys: for log entries from the annotated solver
        3. cont_error_keys: for log entries of the form time step continuity errors

    Returns:
        Dictionary of list of LogKeys
    """
    transport_eqn_keys = build_transport_eqn_keys()

    ogl_annotation_keys = [
        LogKey(search, ["proc", "time"], append_search_to_col=True)
        for search in build_OGLAnnotationKeys(["p"])
    ]

    # time based column name
    col_time = [" [ms]"]

    SolverAnnotationKeys = [
        "MatrixAssemblyU",
        "MomentumPredictor",
        "SolveP",
        "MatrixAssemblyPI:",
        "MatrixAssemblyPII:",
        "TimeStep",
    ]

    foam_annotation_keys = [
        LogKey(search_string=search, columns=col_time, prepend_search_to_col=True)
        for search in SolverAnnotationKeys
    ]

    cont_error = [
        LogKey(
            "time step continuity errors",
            ["ContinuityError " + i for i in ["local", "global", "cumulative"]],
        )
    ]

    return {
        "transp_eqn_keys": transport_eqn_keys,
        "ogl_annotation_keys": ogl_annotation_keys,
        "foam_annotation_keys": foam_annotation_keys,
        "cont_error": cont_error,
    }

def generate_queries() -> list[Query]:
    """This function generates coresponding OBR queries to query the values from the job_documents"""
    log_keys = generate_log_keys()
    queries = []

    # first we generate queries for values from log keys
    for category, log_key_list in log_keys.items():
        for log_key in log_key_list:
            for c in log_key.column_names:
                queries.append(Query(key=c))

    queries = queries + build_default_queries()

    return queries

def build_queries_from_str_list(ls: list[str]) -> list[Query]:
    """Convenience function to build a list of queries from a list of key strings"""
    return list(map(lambda x: Query(key=x), ls))


def to_jobs(path: str) -> list:
    """initialize a list of jobs from a given path"""

    os.chdir(path)

    project = OpenFOAMProject().init_project()
    return [j for j in project]


def grouped_from_query_to_df(
    grouped_jobs: dict[str, list], query: str, index: list
) -> dict:
    """ """
    # TODO detect variations and group them here
    ret = dict
    for group_id, jobs in grouped_jobs.items():
        jobs = filter(lambda x: not x.sp.get("has_child", True), jobs)
        ret[group_id] = query_to_dataframe(jobs, query, index)
    return ret
