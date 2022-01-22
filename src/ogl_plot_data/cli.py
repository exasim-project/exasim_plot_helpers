"""
Module that contains the command line app.

Why does this file exist, and why not put this in __main__?

  You might be tempted to import things from __main__ later, but that will cause
  problems: the code will get executed twice:

  - When you run `python -mogl_plot_data` python will execute
    ``__main__.py`` as a script. That means there won't be any
    ``ogl_plot_data.__main__`` in ``sys.modules``.
  - When you import __main__ it will get executed again (as a module) because
    there's no ``ogl_plot_data.__main__`` in ``sys.modules``.

  Also see (1) from http://click.pocoo.org/5/setuptools/#setuptools-integration
"""
import click

import import_benchmark_data


@click.command()
@click.argument("names", nargs=-1)
def main(names):
    pass


if __name__ == "__main__":

    # by linear solver
    df = import_benchmark_data.import_results("results/cur", "motorcycle_solver")["raw"]
    linear_solver = set(df.index.get_level_values("solver_p"))

    for solver in ["CG", "BiCGStab"]:
        sel = df.loc[solver]
        print(sel["linear_solve_p"] / 1e6)
        sel["linear_solve_p"].plot(kind="bar")

    for solver in ["CG", "BiCGStab"]:
        sel = df.loc[solver]
        print(sel["linear_solve_p"] / sel["linear_solve_U"])

    for solver in ["CG", "BiCGStab"]:
        sel = df.loc[solver]
        print(sel["linear_solve_p"] / sel["deltaT"] / 1e6)
