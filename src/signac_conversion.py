import os
import flow

import obr
import pandas as pd

def build_gko_query(field):
    query = " and ".join(
        [
            f.format(field)
            for f in [
                "solver",
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
        ]
    )

    return query


def build_annotated_query_rel():
    return " and ".join(
        [
            "solver",
            "executor",
            "SolveP_rel",
            "MomentumPredictor_rel",
            "MatrixAssemblyU_rel",
            "MatrixAssemblyPI:_rel",
            "MatrixAssemblyPII:_rel",
            "nCells",
            "nSubDomains",
        ]
    )


def build_annotated_query():
    return " and ".join(
        [
            "solver",
            "executor",
            "SolveP",
            "MomentumPredictor",
            "MatrixAssemblyU",
            "MatrixAssemblyPI:",
            "MatrixAssemblyPII:",
            "TimeStep",
            "nCells",
            "nSubDomains",
        ]
    )



class OpenFOAMProject(flow.FlowProject):
        pass

def to_jobs(path: str) -> list:
    """ initialize a list of jobs from a given path """

    os.chdir(path)

    project = OpenFOAMProject().init_project()
    return [j for j in project]

def from_query_to_df(jobs: list, query: str, index: list):
    """
    """
    res = obr.signac_operations.query_to_dict(jobs, query)
    return pd.DataFrame.from_records([d.result for d in res], index=index).sort_index()


