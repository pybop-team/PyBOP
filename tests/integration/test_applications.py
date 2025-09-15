import warnings

import numpy as np
import pybamm
import pytest

import pybop


class TestApplications:
    """
    A class to test the application methods.
    """

    pytestmark = pytest.mark.integration

    def test_monotonicity_check(self):
        appl = pybop.BaseApplication()

        with pytest.warns(UserWarning, match="OCV is not strictly monotonic."):
            warnings.simplefilter("always")
            appl.check_monotonicity(np.asarray([3, 4, 3]))

    @pytest.fixture
    def parameter_set(self):
        return pybamm.ParameterValues("Chen2020")

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

    def test_interpolant(self, parameter_set, discharge_dataset):
        electrode = "positive"
        parameter_set = pybop.lithium_ion.SPDiffusion.apply_parameter_grouping(
            parameter_set, electrode=electrode
        )
        parameter_set["Electrode OCP [V]"] = pybop.Interpolant(
            discharge_dataset["Stoichiometry"], discharge_dataset["Voltage [V]"]
        )
        model = pybop.lithium_ion.SPDiffusion(
            parameter_set=parameter_set, electrode=electrode, build=True
        )
        solution = model.predict(t_eval=np.linspace(0, 10, 100))
        assert len(solution["Voltage [V]"].data) == 100

    def test_ocp_merge(self, discharge_dataset, charge_dataset):
        ocp_merge = pybop.OCPMerge(
            ocp_discharge=discharge_dataset,
            ocp_charge=charge_dataset,
        )
        merged_dataset = ocp_merge()

        np.testing.assert_allclose(
            merged_dataset["Stoichiometry"][0], discharge_dataset["Stoichiometry"][0]
        )
        np.testing.assert_allclose(
            merged_dataset["Voltage [V]"][0], discharge_dataset["Voltage [V]"][0]
        )
        np.testing.assert_allclose(
            merged_dataset["Stoichiometry"][-1], charge_dataset["Stoichiometry"][0]
        )
        np.testing.assert_allclose(
            merged_dataset["Voltage [V]"][-1], charge_dataset["Voltage [V]"][0]
        )

        # Test with opposite voltage gradient
        discharge_dataset["Voltage [V]"] = np.flipud(discharge_dataset["Voltage [V]"])
        charge_dataset["Voltage [V]"] = np.flipud(charge_dataset["Voltage [V]"])
        ocp_merge = pybop.OCPMerge(
            ocp_discharge=discharge_dataset,
            ocp_charge=charge_dataset,
        )
        merged_dataset = ocp_merge()

        np.testing.assert_allclose(
            merged_dataset["Stoichiometry"][0], charge_dataset["Stoichiometry"][-1]
        )
        np.testing.assert_allclose(
            merged_dataset["Voltage [V]"][0], charge_dataset["Voltage [V]"][-1]
        )
        np.testing.assert_allclose(
            merged_dataset["Stoichiometry"][-1],
            discharge_dataset["Stoichiometry"][-1],
        )
        np.testing.assert_allclose(
            merged_dataset["Voltage [V]"][-1], discharge_dataset["Voltage [V]"][-1]
        )

    def test_ocp_average(self, discharge_dataset, charge_dataset):
        for allow_stretching in [True, False]:
            # Estimate the shift and generate the average open-circuit potential
            ocp_average = pybop.OCPAverage(
                ocp_discharge=discharge_dataset,
                ocp_charge=charge_dataset,
                allow_stretching=allow_stretching,
            )
            ocp_average()

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
        ocv_fit()

        np.testing.assert_allclose(ocv_fit.stretch, nom_capacity, rtol=2e-3, atol=2e-3)
        np.testing.assert_allclose(
            ocv_fit.shift, 0.1 * nom_capacity, rtol=5e-3, atol=5e-3
        )

    @pytest.fixture
    def half_cell_model(self):
        parameter_set = pybamm.ParameterValues("Xu2019")
        return pybop.lithium_ion.SPMe(
            parameter_set=parameter_set, options={"working electrode": "positive"}
        )

    @pytest.fixture
    def pulse_data(self, half_cell_model):
        sigma = 5e-4
        initial_state = {"Initial SoC": 0.9}
        experiment = pybamm.Experiment(
            [
                "Rest for 1 second",
                "Discharge at 2C for 5 minutes (10 second period)",
                "Rest for 15 minutes (10 second period)",
            ]
        )
        values = half_cell_model.predict(
            initial_state=initial_state, experiment=experiment
        )
        corrupt_values = values["Voltage [V]"].data + np.random.normal(
            0, sigma, len(values["Voltage [V]"].data)
        )
        start = np.where(values["Time [s]"].data == 1)[0][0] - 1
        return pybop.Dataset(
            {
                "Time [s]": values["Time [s]"].data[start:],
                "Current function [A]": values["Current [A]"].data[start:],
                "Discharge capacity [A.h]": values["Discharge capacity [A.h]"].data[
                    start:
                ],
                "Voltage [V]": corrupt_values[start:],
            }
        )

    def test_gitt_pulse_fit(self, half_cell_model, pulse_data):
        parameter_set = pybop.lithium_ion.SPDiffusion.apply_parameter_grouping(
            half_cell_model.parameter_set, electrode="positive"
        )
        diffusion_time = parameter_set["Particle diffusion time scale [s]"]

        gitt_fit = pybop.GITTPulseFit(parameter_set=parameter_set, electrode="positive")
        gitt_results = gitt_fit(gitt_pulse=pulse_data)

        np.testing.assert_allclose(gitt_results.x[0], diffusion_time, rtol=5e-2)

    def test_gitt_fit(self, half_cell_model, pulse_data):
        parameter_set = pybop.lithium_ion.SPDiffusion.apply_parameter_grouping(
            half_cell_model.parameter_set, electrode="positive"
        )
        diffusion_time = parameter_set["Particle diffusion time scale [s]"]

        with pytest.raises(
            ValueError, match="The initial current in the pulse dataset must be zero."
        ):
            gitt_fit = pybop.GITTFit(
                gitt_dataset=pulse_data,
                pulse_index=[np.arange(2, len(pulse_data["Current function [A]"]))],
                parameter_set=parameter_set,
            )
            gitt_fit()

        gitt_fit = pybop.GITTFit(
            gitt_dataset=pulse_data,
            pulse_index=[np.arange(len(pulse_data["Current function [A]"]))],
            parameter_set=parameter_set,
            electrode="positive",
        )
        gitt_parameter_data = gitt_fit()

        np.testing.assert_allclose(
            gitt_parameter_data["Particle diffusion time scale [s]"],
            np.asarray([diffusion_time]),
            rtol=5e-2,
        )
        assert gitt_parameter_data["Root Mean Squared Error [V]"][0] < 2e-3
