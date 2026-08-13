import sys
from copy import copy
from types import SimpleNamespace

import numpy as np
import polars as pl
import pybamm
import pytest
from pybamm.models.full_battery_models.lithium_ion.electrode_soh import (
    get_min_max_stoichiometries,
)

import pybop
from pybop.applications.utils import (
    OpenCircuitVoltage,
    filter_with_preceding_row,
    get_cells,
    get_ocp_functions,
    get_ocv_function,
    make_voltage_monotonic,
    shift_ocv_to,
)


class TestUtils:
    """
    A class to test the utilities for battery applications.
    """

    pytestmark = pytest.mark.unit

    @pytest.fixture
    def parameter_values(self):
        param = pybamm.ParameterValues("Chen2020")
        x_0, x_100, y_100, y_0 = get_min_max_stoichiometries(param)
        param["Minimum negative stoichiometry"] = x_0
        param["Maximum negative stoichiometry"] = x_100
        param["Minimum positive stoichiometry"] = y_100
        param["Maximum positive stoichiometry"] = y_0

        sto = np.linspace(0, 1, 101)
        param["Positive electrode OCP [V]"] = pybop.Interpolant(
            sto, param.evaluate(param["Positive electrode OCP [V]"](pybamm.Vector(sto)))
        )
        param["Negative electrode OCP [V]"] = pybop.Interpolant(
            sto, param.evaluate(param["Negative electrode OCP [V]"](pybamm.Vector(sto)))
        )
        return param

    def test_get_ocp_functions(self, parameter_values):
        positive_ocp_function, negative_ocp_function = get_ocp_functions(
            parameter_values, OCP_type="OCP"
        )
        sto = np.linspace(0, 1, 5)
        assert np.all(positive_ocp_function(sto) > 0)
        assert np.all(negative_ocp_function(sto) > 0)

    def test_open_circuit_voltage(self, parameter_values):
        positive_ocp_function, negative_ocp_function = get_ocp_functions(
            parameter_values, OCP_type="OCP"
        )
        ocv_function = OpenCircuitVoltage(
            positive_ocp_function,
            parameter_values["Maximum positive stoichiometry"],
            parameter_values["Minimum positive stoichiometry"],
            negative_ocp_function,
            parameter_values["Minimum negative stoichiometry"],
            parameter_values["Maximum negative stoichiometry"],
        )
        soc = np.linspace(0, 1, 5)
        assert np.all(ocv_function(soc) > 0)

    def test_get_ocv_function(self, parameter_values):
        ocv_function = get_ocv_function(parameter_values, OCP_type="OCP")
        soc = np.linspace(0, 1, 5)
        assert np.all(ocv_function(soc) > 0)

    def test_shift_ocv_to(self, parameter_values):
        ocv_function = get_ocv_function(parameter_values, OCP_type="OCP")
        soc = np.linspace(0, 1, 5)
        voltage_points = ocv_function(soc) + 0.1
        param, soc_points = shift_ocv_to(voltage_points, parameter_values, soc)
        ocv_function = get_ocv_function(param)
        assert np.all(ocv_function(soc) > 0.1)
        np.testing.assert_allclose(soc, soc_points, atol=2.5e-4)

    def test_make_voltage_monotonic(self):
        # Test ascending
        data = SimpleNamespace()
        voltage = np.concatenate(([0], np.random.rand(99), [1]))
        data.lf = pl.LazyFrame({"Voltage [V]": copy(voltage)})
        data = make_voltage_monotonic(data)
        assert np.all(data.lf["Voltage [V]"].to_numpy() <= voltage)

        # Test descending
        data = SimpleNamespace()
        voltage = np.concatenate(([1], np.random.rand(99), [0]))
        data.lf = pl.LazyFrame({"Voltage [V]": copy(voltage)})
        data = make_voltage_monotonic(data)
        assert np.all(data.lf["Voltage [V]"].to_numpy() >= voltage)

    @pytest.mark.skipif(
        sys.version_info < (3, 11), reason="requires a python version >= 3.11"
    )
    @pytest.mark.skipif(
        sys.version_info >= (3, 13), reason="requires a python version < 3.13"
    )
    def test_get_cells_and_filter(self):
        import pyprobe

        cells = get_cells()
        assert isinstance(cells, list)
        cell = cells[0]
        assert isinstance(cell, pyprobe.Cell)

        cycle = cell.procedure["pOCV"].experiment("Pseudo OCV").cycle(0)
        charge_dataset = pybop.import_pyprobe_result(cycle.charge().step(0))
        dataset = pybop.import_pyprobe_result(
            filter_with_preceding_row(
                cell.procedure["pOCV"],
                experiment="Pseudo OCV",
                cycle=0,
                phase="charge",
                step=0,
            )
        )
        assert len(dataset) == len(charge_dataset) + 1

        discharge_dataset = pybop.import_pyprobe_result(cycle.discharge().step(0))
        dataset = pybop.import_pyprobe_result(
            filter_with_preceding_row(
                cell.procedure["pOCV"],
                experiment="Pseudo OCV",
                cycle=0,
                phase="discharge",
                step=0,
            )
        )
        assert len(dataset) == len(discharge_dataset) + 1

        rest_dataset = pybop.import_pyprobe_result(cycle.rest().step(0))
        dataset = pybop.import_pyprobe_result(
            filter_with_preceding_row(
                cell.procedure["pOCV"],
                experiment="Pseudo OCV",
                cycle=0,
                phase="rest",
                step=0,
            )
        )
        assert len(dataset) == len(rest_dataset) + 1
