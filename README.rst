========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |github-actions| |requires|
        | |codecov|
    * - package
      - | |version| |wheel| |supported-versions| |supported-implementations|
        | |commits-since|
.. |docs| image:: https://readthedocs.org/projects/ogl_plot_data/badge/?style=flat
    :target: https://ogl_plot_data.readthedocs.io/
    :alt: Documentation Status

.. |github-actions| image:: https://github.com/ogl_plot_data/ogl_plot_data/actions/workflows/github-actions.yml/badge.svg
    :alt: GitHub Actions Build Status
    :target: https://github.com/ogl_plot_data/ogl_plot_data/actions

.. |requires| image:: https://requires.io/github/ogl_plot_data/ogl_plot_data/requirements.svg?branch=main
    :alt: Requirements Status
    :target: https://requires.io/github/ogl_plot_data/ogl_plot_data/requirements/?branch=main

.. |codecov| image:: https://codecov.io/gh/ogl_plot_data/ogl_plot_data/branch/main/graphs/badge.svg?branch=main
    :alt: Coverage Status
    :target: https://codecov.io/github/ogl_plot_data/ogl_plot_data

.. |version| image:: https://img.shields.io/pypi/v/ogl-plot-data.svg
    :alt: PyPI Package latest release
    :target: https://pypi.org/project/ogl-plot-data

.. |wheel| image:: https://img.shields.io/pypi/wheel/ogl-plot-data.svg
    :alt: PyPI Wheel
    :target: https://pypi.org/project/ogl-plot-data

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/ogl-plot-data.svg
    :alt: Supported versions
    :target: https://pypi.org/project/ogl-plot-data

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/ogl-plot-data.svg
    :alt: Supported implementations
    :target: https://pypi.org/project/ogl-plot-data

.. |commits-since| image:: https://img.shields.io/github/commits-since/ogl_plot_data/ogl_plot_data/v0.0.0.svg
    :alt: Commits since latest release
    :target: https://github.com/ogl_plot_data/ogl_plot_data/compare/v0.0.0...main



.. end-badges

plotting ogl benchmark results

* Free software: BSD 2-Clause License

Installation
============

::

    pip install ogl-plot-data

You can also install the in-development version with::

    pip install https://github.com/ogl_plot_data/ogl_plot_data/archive/main.zip


Documentation
=============


https://ogl_plot_data.readthedocs.io/


Development
===========

To run all the tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
