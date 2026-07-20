import os
import runpy
import sys

import pytest

import pybop


class TestExamples:
    """
    A class to test the example scripts.
    """

    def list_of_examples():
        examples_list = []
        path_to_example_scripts = os.path.join(
            pybop.script_path, "..", "examples", "scripts"
        )
        for dirpath, _, filenames in os.walk(path_to_example_scripts):
            for file in filenames:
                if file.endswith(".py"):
                    examples_list.append(os.path.join(dirpath, file))
        return examples_list

    @pytest.mark.parametrize("example", list_of_examples())
    @pytest.mark.examples
    def test_example_scripts(self, example):
        if (
            sys.version_info >= (3, 13)
            and os.path.basename(example) == "bayesian_feature_fitting.py"
        ):
            pytest.skip("This example requires a python version < 3.13")
        elif (
            sys.version_info < (3, 11)
            and os.path.basename(example) == "generate_synthetic_data.py"
        ):
            pytest.skip("This example requires a python version >= 3.11")
        elif (
            sys.version_info >= (3, 13)
            and os.path.basename(example) == "generate_synthetic_data.py"
        ):
            pytest.skip("This example requires a python version < 3.13")
        else:
            runpy.run_path(example)
