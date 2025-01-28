import numpy as np
import pytest

import pybop


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

    def test_ocp_average(self, parameter_set):
        ocp_function = parameter_set["Positive electrode OCP [V]"]

        discharge_sto = np.linspace(0, 0.9, 91)
        discharge_voltage = ocp_function(discharge_sto + 0.02) + self.noise(1e-3, 91)
        charge_sto = np.linspace(1, 0.1, 91)
        charge_voltage = ocp_function(charge_sto - 0.02) + self.noise(1e-3, 91)

        # Create the charge and discharge datasets
        discharge_dataset = pybop.Dataset(
            {"Stoichiometry": discharge_sto, "Voltage [V]": discharge_voltage}
        )
        charge_dataset = pybop.Dataset(
            {"Stoichiometry": charge_sto, "Voltage [V]": charge_voltage}
        )

        for allow_stretching in [True, False]:
            # Estimate the shift and generate the average open-circuit potential
            ocp_average = pybop.ocp_average(
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
            {"Charge capacity [A.h]": (sto + 0.1) * nom_capacity, "Voltage [V]": voltage}
        )

        # Estimate the stoichiometry corresponding to the GITT-OCV
        ocv_fit = pybop.stoichiometric_fit(
            ocv_dataset=ocv_dataset,
            ocv_function=ocv_function,
        )

        np.testing.assert_allclose(ocv_fit.stretch, nom_capacity, rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(ocv_fit.shift, 0.1*nom_capacity, rtol=1e-3, atol=1e-3)
