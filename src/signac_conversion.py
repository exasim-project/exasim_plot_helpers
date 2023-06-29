import os

import flow
import pandas as pd
from obr.core.queries import Query, query_to_dataframe


def build_gko_query(field):
    l = list(
        map(
            lambda x: Query(key=x),
            [
                f.format(field)
                for f in [
                    "solver",
                    "executor",
                    "preconditioner",
                    "{}: update_local_matrix_data:",
                    "{}: update_non_local_matrix_data:",
                    "{}_matrix: call_update:",
                    "{}_rhs: call_update:",
                    "{}: init_precond:",
                    "{}: generate_solver:",
                    "{}: solve:",
                    "{}: copy_x_back:",
                    "nCells",
                    "nSubDomains",
                ]
            ],
        )
    )
    l.append(Query(key="completed", value=True))
    return l


def build_annotated_query() -> list:
    l = list(
        map(
            lambda x: Query(key=x),
            [
                "solver",
                "host",
                "timestamp",
                "preconditioner",
                "executor",
                "SolveP",
                "MomentumPredictor",
                "MatrixAssemblyU",
                "MatrixAssemblyPI:",
                "MatrixAssemblyPII:",
                "TimeStep",
                "nCells",
                "nSubDomains",
                "iter_p",
            ],
        )
    )
    # l.append(Query(key="completed", value=True))
    return l


def build_annotated_query_from_list(ls: list) -> list:
    l = list(map(lambda x: Query(key=x), ls))
    return l


class OpenFOAMProject(flow.FlowProject):
    pass


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
