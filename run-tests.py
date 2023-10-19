#!/usr/bin/env python3
#
# Runs all unit tests included in PyBOP.
#
# This file is adapted from PINTS (https://github.com/pints-team/pints/).
#
import argparse
import datetime
import os
import re
import subprocess
import sys
import unittest


def run_unit_tests():
    """
    Runs unit tests (without subprocesses).
    """
    # tests = os.path.join('pybop', 'tests')
    tests = os.path.join(os.path.dirname(__file__), 'tests', 'unit')
    suite = unittest.defaultTestLoader.discover(tests, pattern='test*.py')
    res = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if res.wasSuccessful() else 1)


# def run_flake8():

# def run_copyright_checks():

# def run_doctests():

# def doctest_sphinx():

# def doctest_examples_readme():

# def doctest_rst_and_public_interface():

# def check_exposed_symbols(module, submodule_names, doc_symbols):

# def get_all_documented_symbols():

# def run_notebook_tests():

# def run_notebook_interfaces_tests():

# def list_notebooks(root, recursive=True, ignore_list=None, notebooks=None):

# def test_notebook(path):

# def export_notebook(ipath, opath):


if __name__ == '__main__':
    # Prevent CI from hanging on multiprocessing tests
    import multiprocessing
    multiprocessing.set_start_method('spawn')

    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description='Run unit tests for PyBOP.',
        epilog='To run individual unit tests, use e.g.'
               ' $ python tests/unit/test_parameterisation.py',
    )
    # Unit tests
    parser.add_argument(
        '--unit',
        action='store_true',
        help='Run all unit tests using the `python` interpreter.',
    )
#     # Notebook tests
#     parser.add_argument(
#         '--books',
#         action='store_true',
#         help='Test only the fast Jupyter notebooks in `examples`.',
#     )
#     parser.add_argument(
#         '--interfaces',
#         action='store_true',
#         help='Test only the fast Jupyter notebooks in `examples/interfaces`.',
#     )
#     parser.add_argument(
#         '-debook',
#         nargs=2,
#         metavar=('in', 'out'),
#         help='Export a Jupyter notebook to a Python file for manual testing.',
#     )
#     # Doctests
#     parser.add_argument(
#         '--doctest',
#         action='store_true',
#         help='Run any doctests, check if docs can be built',
#     )
#     # Copyright checks
#     parser.add_argument(
#         '--copyright',
#         action='store_true',
#         help='Check copyright runs to the current year',
#     )
    # Combined test sets
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick checks (unit tests)', #, flake8, docs)',
    )

    # Parse!
    args = parser.parse_args()

    # Run tests
    has_run = False
    # Unit tests
    if args.unit:
        has_run = True
        run_unit_tests()
#     # Doctests
#     if args.doctest:
#         has_run = True
#         run_doctests()
#     # Copyright checks
#     if args.copyright:
#         has_run = True
#         run_copyright_checks()
#     # Notebook tests
#     elif args.books:
#         has_run = True
#         run_notebook_tests()
#     if args.interfaces:
#         has_run = True
#         run_notebook_interfaces_tests()
#     if args.debook:
#         has_run = True
#         export_notebook(*args.debook)
    # Combined test sets
    if args.quick:
        has_run = True
#         run_flake8()
        run_unit_tests()
#         run_doctests()
    # Help
    if not has_run:
        parser.print_help()
