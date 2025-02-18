import numpy as np
import pytest

import pybop
from pybop.models.lithium_ion.basic_SP_diffusion import (
    convert_physical_to_electrode_parameters,
)


class TestApplications:
    """
    A class to test the application methods.
    """

    pytestmark = pytest.mark.integration

    @pytest.fixture
    def parameter_set(self):
        return pybop.ParameterSet("Chen2020")

    def noise(self, sigma, values):
        return np.random.normal(0, sigma, values)

    @pytest.fixture
    def discharge_dataset(self, parameter_set):
        ocp_function = parameter_set["Positive electrode OCP [V]"]

        discharge_sto = np.linspace(0, 0.9, 91)
        discharge_voltage = ocp_function(discharge_sto + 0.02) + self.noise(1e-3, 91)

        return pybop.Dataset(
            {"Stoichiometry": discharge_sto, "Voltage [V]": discharge_voltage}
        )

    @pytest.fixture
    def charge_dataset(self, parameter_set):
        ocp_function = parameter_set["Positive electrode OCP [V]"]

        charge_sto = np.linspace(1, 0.1, 91)
        charge_voltage = ocp_function(charge_sto - 0.02) + self.noise(1e-3, 91)

        return pybop.Dataset(
            {"Stoichiometry": charge_sto, "Voltage [V]": charge_voltage}
        )

    def test_ocp_blend(self, discharge_dataset, charge_dataset):
        ocp_blend = pybop.OCPBlend(
            ocp_discharge=discharge_dataset,
            ocp_charge=charge_dataset,
        )

        np.testing.assert_allclose(
            ocp_blend.dataset["Stoichiometry"][0], discharge_dataset["Stoichiometry"][0]
        )
        np.testing.assert_allclose(
            ocp_blend.dataset["Voltage [V]"][0], discharge_dataset["Voltage [V]"][0]
        )
        np.testing.assert_allclose(
            ocp_blend.dataset["Stoichiometry"][-1], charge_dataset["Stoichiometry"][0]
        )
        np.testing.assert_allclose(
            ocp_blend.dataset["Voltage [V]"][-1], charge_dataset["Voltage [V]"][0]
        )

        # Test with opposite voltage gradient
        discharge_dataset["Voltage [V]"] = np.flipud(discharge_dataset["Voltage [V]"])
        charge_dataset["Voltage [V]"] = np.flipud(charge_dataset["Voltage [V]"])
        ocp_blend = pybop.OCPBlend(
            ocp_discharge=discharge_dataset,
            ocp_charge=charge_dataset,
        )

        np.testing.assert_allclose(
            ocp_blend.dataset["Stoichiometry"][0], charge_dataset["Stoichiometry"][-1]
        )
        np.testing.assert_allclose(
            ocp_blend.dataset["Voltage [V]"][0], charge_dataset["Voltage [V]"][-1]
        )
        np.testing.assert_allclose(
            ocp_blend.dataset["Stoichiometry"][-1],
            discharge_dataset["Stoichiometry"][-1],
        )
        np.testing.assert_allclose(
            ocp_blend.dataset["Voltage [V]"][-1], discharge_dataset["Voltage [V]"][-1]
        )

    def test_ocp_average(self, discharge_dataset, charge_dataset):
        for allow_stretching in [True, False]:
            # Estimate the shift and generate the average open-circuit potential
            ocp_average = pybop.OCPAverage(
                ocp_discharge=discharge_dataset,
                ocp_charge=charge_dataset,
                allow_stretching=allow_stretching,
            )

            np.testing.assert_allclose(ocp_average.stretch, 1.0, rtol=1e-3, atol=1e-3)
            np.testing.assert_allclose(ocp_average.shift, 0.02, rtol=1e-3, atol=1e-3)

    def test_stoichiometry_fit(self, parameter_set):
        ocv_function = parameter_set["Positive electrode OCP [V]"]
        nom_capacity = parameter_set["Nominal cell capacity [A.h]"]

        sto = np.linspace(0, 0.9, 91)
        voltage = ocv_function(sto) + self.noise(2e-3, 91)

        # Create the OCV dataset
        ocv_dataset = pybop.Dataset(
            {
                "Charge capacity [A.h]": (sto + 0.1) * nom_capacity,
                "Voltage [V]": voltage,
            }
        )

        # Estimate the stoichiometry corresponding to the GITT-OCV
        ocv_fit = pybop.OCPCapacityToStoichiometry(
            ocv_dataset=ocv_dataset,
            ocv_function=ocv_function,
        )

        np.testing.assert_allclose(ocv_fit.stretch, nom_capacity, rtol=2e-3, atol=2e-3)
        np.testing.assert_allclose(
            ocv_fit.shift, 0.1 * nom_capacity, rtol=5e-3, atol=5e-3
        )

    @pytest.fixture
    def half_cell_model(self):
        parameter_set = pybop.ParameterSet("Xu2019")
        return pybop.lithium_ion.SPMe(
            parameter_set=parameter_set, options={"working electrode": "positive"}
        )

    @pytest.fixture
    def pulse_data(self, half_cell_model):
        sigma = 1e-3
        initial_state = {"Initial SoC": 0.9}
        experiment = pybop.Experiment(
            [
                "Rest for 1 second",
                "Discharge at 1C for 10 minutes (10 second period)",
                "Rest for 20 minutes",
            ]
        )
        values = half_cell_model.predict(
            initial_state=initial_state, experiment=experiment
        )
        corrupt_values = values["Voltage [V]"].data + np.random.normal(
            0, sigma, len(values["Voltage [V]"].data)
        )
        return pybop.Dataset(
            {
                "Time [s]": values["Time [s]"].data,
                "Current function [A]": values["Current [A]"].data,
                "Discharge capacity [A.h]": values["Discharge capacity [A.h]"].data,
                "Voltage [V]": corrupt_values,
            }
        )

    def test_gitt_pulse_fit(self, half_cell_model, pulse_data):
        parameter_set = convert_physical_to_electrode_parameters(
            half_cell_model.parameter_set, "positive"
        )
        diffusion_time = parameter_set["Particle diffusion time scale [s]"]

        gitt_fit = pybop.GITTPulseFit(pulse_data, parameter_set)

        np.testing.assert_allclose(gitt_fit.results.x[0], diffusion_time, rtol=5e-2)
