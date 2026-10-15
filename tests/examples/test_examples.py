import runpy
import shutil
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path("examples") / "scripts"


class TestExamples:
    """
    A class to test the example scripts.
    """

    pytestmark = pytest.mark.examples

    def list_of_example_scripts() -> list[Path]:
        return [
            p
            for p in SCRIPTS_DIR.rglob("*.py")
            if not str(p.parent).endswith("dfn_parameterisation")
        ]

    @pytest.fixture
    def list_of_pipeline_scripts(self) -> list[Path]:
        path_to_pipeline = SCRIPTS_DIR / "dfn_parameterisation"
        return list(path_to_pipeline.glob("*.py"))

    @pytest.mark.parametrize(
        "example", list_of_example_scripts(), ids=lambda val: f"{val.name}"
    )
    def test_example_scripts(self, example: Path, tmp_path):
        v = sys.version_info
        if v >= (3, 13) and example.name == "bayesian_feature_fitting.py":
            pytest.skip("This example requires a python version < 3.13")
        elif v < (3, 11) and example.name == "generate_synthetic_data.py":
            pytest.skip("This example requires a python version >= 3.11")
        elif v >= (3, 13) and example.name == "generate_synthetic_data.py":
            pytest.skip("This example requires a python version < 3.13")
        else:
            # Copy the example to the temporary directory before running it
            new_example_path = tmp_path / example.name
            shutil.copy(example, new_example_path)
            runpy.run_path(new_example_path)

    @pytest.mark.skipif(
        sys.version_info < (3, 11), reason="requires a python version >= 3.11"
    )
    @pytest.mark.skipif(
        sys.version_info >= (3, 13), reason="requires a python version < 3.13"
    )
    def test_pipeline_scripts(self, list_of_pipeline_scripts: list[Path], tmp_path):
        # The pipeline scripts must be run sequentially
        for example in sorted(list_of_pipeline_scripts):
            new_example_path = tmp_path / example.name
            shutil.copy(example, new_example_path)
            runpy.run_path(new_example_path)
