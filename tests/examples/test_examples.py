import pybop
import pytest
import runpy
import os


class TestExamples:
    """
    A class to test the example scripts.
    """

    def list_of_examples():
        list = []
        path_to_example_scripts = os.path.join(
            pybop.script_path, "..", "examples", "scripts"
        )
        for example in os.listdir(path_to_example_scripts):
            if example.endswith(".py"):
                list.append(os.path.join(path_to_example_scripts, example))
        return list

    @pytest.mark.parametrize("example", list_of_examples())
    @pytest.mark.examples
    def test_example_scripts(self, example):
        runpy.run_path(example)
