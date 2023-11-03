import pybop
import pytest
import runpy
import os


class TestExamples:
    """
    A class to test the example scripts.
    """

    @pytest.mark.unit
    def test_example_scripts(self):
        path_to_example_scripts = os.path.join(
            pybop.script_path, "..", "examples", "scripts"
        )
        for example in os.listdir(path_to_example_scripts):
            if example.endswith(".py"):
                runpy.run_path(os.path.join(path_to_example_scripts, example))
