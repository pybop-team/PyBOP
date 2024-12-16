import os
import runpy

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
        runpy.run_path(example)
