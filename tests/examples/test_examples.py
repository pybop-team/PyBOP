import os
import runpy
import sys

import pytest

import pybop


class TestExamples:
    """
    A class to test the example scripts.
    """

    pytestmark = pytest.mark.examples

    def list_of_example_scripts():
        examples_list = []
        path_to_example_scripts = os.path.join(
            pybop.script_path, "..", "examples", "scripts"
        )
        for dirpath, _, filenames in os.walk(path_to_example_scripts):
            if not dirpath.endswith("dfn_parameterisation"):
                for file in filenames:
                    if file.endswith(".py"):
                        examples_list.append(os.path.join(dirpath, file))
        return examples_list

    @pytest.fixture
    def list_of_pipeline_scripts(self):
        # The pipeline scripts must be run sequentially
        examples_list = []
        path_to_example_scripts = os.path.join(
            pybop.script_path, "..", "examples", "scripts", "dfn_parameterisation"
        )
        for dirpath, _, filenames in os.walk(path_to_example_scripts):
            for file in filenames:
                if file.endswith(".py"):
                    examples_list.append(os.path.join(dirpath, file))
        return examples_list

    @pytest.mark.parametrize("example", list_of_example_scripts())
    def test_example_scripts(self, example):
        if (
            sys.version_info >= (3, 13)
            and os.path.basename(example) == "bayesian_feature_fitting.py"
        ):
            pytest.skip("This example requires a python version < 3.13")
        else:
            runpy.run_path(example)

    @pytest.mark.skipif(
        sys.version_info < (3, 11), reason="requires a python version >= 3.11"
    )
    @pytest.mark.skipif(
        sys.version_info >= (3, 13), reason="requires a python version < 3.13"
    )
    def test_pipeline_scripts(self, list_of_pipeline_scripts):
        # The pipeline scripts must be run sequentially
        for example in sorted(list_of_pipeline_scripts):
            runpy.run_path(example)
