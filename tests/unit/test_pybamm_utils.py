import sys
from pathlib import Path

import pybamm
import pytest

import pybop

PROCEDURE_DIR = Path("examples") / "scripts/synthetic_data/procedures"


@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="requires a python version >= 3.11"
)
@pytest.mark.skipif(
    sys.version_info >= (3, 13), reason="requires a python version < 3.13"
)
class TestPybammUtils:
    """
    A class to test the synthetic generation procedure.
    """

    pytestmark = pytest.mark.unit

    def test_simulate_procedure(self, tmp_path):
        import pyprobe

        model = pybamm.lithium_ion.SPM()
        full_cell_parameters = pybamm.ParameterValues("Chen2020")
        cell_info = {
            "Cell type": "LG M50 Synthetic",
            "Cell format": "full cell",
        }
        procedures = [
            "Capacity determination.json",
            "pOCV.json",
            "GITT.json",
            "EIS charge.json",
            "Validation cycling.json",
            "Validation different SOC.json",
            "Validation pulses.json",
        ]
        cell = pybop.pybamm.simulate_procedure(
            info=cell_info,
            model=model,
            parameter_values=full_cell_parameters,
            spec_path=[PROCEDURE_DIR / p for p in procedures],
        )
        assert isinstance(cell, pyprobe.Cell)

        # Test export
        pybop.pybamm.archive_data(cell=cell, archive_root=tmp_path)

    def test_convert_to_half_cell_parameters(self):
        import pyprobe

        model_options = {
            "working electrode": "positive"
        }  # PyBaMM uses "positive" for all half-cells
        half_cell_model = pybamm.lithium_ion.SPM(model_options)
        full_cell_parameters = pybamm.ParameterValues("Chen2020")

        for electrode in ["negative", "positive"]:
            half_cell_parameters = pybop.pybamm.convert_to_half_cell_parameters(
                full_cell_parameters, electrode
            )
            cell_info = {
                "Cell type": "LG M50 Synthetic",
                "Cell format": f"{electrode} half cell",
            }
            cell = pybop.pybamm.simulate_procedure(
                info=cell_info,
                model=half_cell_model,
                parameter_values=half_cell_parameters,
                spec_path=PROCEDURE_DIR / f"pOCP {electrode}.json",
            )
            assert isinstance(cell, pyprobe.Cell)

    def test_input_validation(self):
        model = pybamm.lithium_ion.SPM()
        full_cell_parameters = pybamm.ParameterValues("Chen2020")

        cell_info = {
            "Cell type": "LG M50 Synthetic",
            "Cell format": "unknown",
        }
        with pytest.raises(ValueError, match="Unsupported cell type"):
            pybop.pybamm.simulate_procedure(
                info=cell_info,
                model=model,
                parameter_values=full_cell_parameters,
                spec_path=PROCEDURE_DIR / "pOCV.json",
            )

        cell_info = {
            "Cell type": "LG M50 Synthetic",
            "Cell format": "half cell negative",
        }
        full_cell_parameters.pop("Negative electrode OCP [V]")
        with pytest.raises(KeyError, match="source parameter was not found"):
            pybop.pybamm.convert_to_half_cell_parameters(
                full_cell_parameters, "negative"
            )
