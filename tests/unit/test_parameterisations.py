import pybop
import pybamm
import pytest
import numpy as np


class TestModelParameterisation:
    """
    A class to test the model parameterisation methods.
    """

    @pytest.mark.parametrize("init_soc", [0.3, 0.5, 0.8])
    @pytest.mark.unit
    def test_spm(self, init_soc):
        # Define model
        parameter_set = pybop.ParameterSet("pybamm", "Chen2020")
        model = pybop.lithium_ion.SPM(parameter_set=parameter_set)

        # Form observations
        x0 = np.array([0.52, 0.63])
        solution = self.getdata(model, x0, init_soc)

        observations = [
            pybop.Observed("Time [s]", solution["Time [s]"].data),
            pybop.Observed("Current function [A]", solution["Current [A]"].data),
            pybop.Observed("Voltage [V]", solution["Terminal voltage [V]"].data),
        ]

        # Fitting parameters
        params = [
            pybop.Parameter(
                "Negative electrode active material volume fraction",
                prior=pybop.Gaussian(0.5, 0.02),
                bounds=[0.375, 0.625],
            ),
            pybop.Parameter(
                "Positive electrode active material volume fraction",
                prior=pybop.Gaussian(0.65, 0.02),
                bounds=[0.525, 0.75],
            ),
        ]

        parameterisation = pybop.Optimisation(
            model, observations=observations, fit_parameters=params, init_soc=init_soc
        )

        # get RMSE estimate using NLOpt
        results, last_optim, num_evals = parameterisation.rmse(
            signal="Voltage [V]", method="nlopt"
        )

        # Assertions (for testing purposes only)
        np.testing.assert_allclose(last_optim, 0, atol=1e-2)
        np.testing.assert_allclose(results, x0, atol=5e-2)

    @pytest.mark.parametrize("init_soc", [0.3, 0.5, 0.8])
    @pytest.mark.unit
    def test_spme(self, init_soc):
        # Define model
        parameter_set = pybop.ParameterSet("pybamm", "Chen2020")
        model = pybop.lithium_ion.SPMe(parameter_set=parameter_set)

        # Form observations
        x0 = np.array([0.52, 0.63])
        solution = self.getdata(model, x0, init_soc)

        observations = [
            pybop.Observed("Time [s]", solution["Time [s]"].data),
            pybop.Observed("Current function [A]", solution["Current [A]"].data),
            pybop.Observed("Voltage [V]", solution["Terminal voltage [V]"].data),
        ]

        # Fitting parameters
        params = [
            pybop.Parameter(
                "Negative electrode active material volume fraction",
                prior=pybop.Gaussian(0.5, 0.02),
                bounds=[0.375, 0.625],
            ),
            pybop.Parameter(
                "Positive electrode active material volume fraction",
                prior=pybop.Gaussian(0.65, 0.02),
                bounds=[0.525, 0.75],
            ),
        ]

        parameterisation = pybop.Optimisation(
            model, observations=observations, fit_parameters=params, init_soc=init_soc
        )

        # get RMSE estimate using NLOpt
        results, last_optim, num_evals = parameterisation.rmse(
            signal="Voltage [V]", method="nlopt"
        )

        # Assertions (for testing purposes only)
        np.testing.assert_allclose(last_optim, 0, atol=1e-2)
        np.testing.assert_allclose(results, x0, rtol=5e-2)

    @pytest.mark.parametrize("init_soc", [0.3, 0.5, 0.8])
    @pytest.mark.unit
    def test_model_misparameterisation(self, init_soc):
        # Define model
        parameter_set = pybop.ParameterSet("pybamm", "Chen2020")
        model = pybop.lithium_ion.SPM(parameter_set=parameter_set)

        second_parameter_set = pybop.ParameterSet("pybamm", "Ecker2015")
        second_model = pybop.lithium_ion.SPM(parameter_set=second_parameter_set)

        # Form observations
        x0 = np.array([0.52, 0.63])
        solution = self.getdata(second_model, x0, init_soc)

        observations = [
            pybop.Observed("Time [s]", solution["Time [s]"].data),
            pybop.Observed("Current function [A]", solution["Current [A]"].data),
            pybop.Observed("Voltage [V]", solution["Terminal voltage [V]"].data),
        ]

        # Fitting parameters
        params = [
            pybop.Parameter(
                "Negative electrode active material volume fraction",
                prior=pybop.Gaussian(0.5, 0.02),
                bounds=[0.375, 0.625],
            ),
            pybop.Parameter(
                "Positive electrode active material volume fraction",
                prior=pybop.Gaussian(0.65, 0.02),
                bounds=[0.525, 0.75],
            ),
        ]

        parameterisation = pybop.Optimisation(
            model, observations=observations, fit_parameters=params, init_soc=init_soc
        )

        # get RMSE estimate using NLOpt
        results, last_optim, num_evals = parameterisation.rmse(
            signal="Voltage [V]", method="nlopt"
        )

        # Assertions (for testing purposes only)
        with np.testing.assert_raises(AssertionError):
            np.testing.assert_allclose(last_optim, 0, atol=1e-2)
            np.testing.assert_allclose(results, x0, rtol=5e-1)

    def getdata(self, model, x0, init_soc):
        model.parameter_set.update(
            {
                "Negative electrode active material volume fraction": x0[0],
                "Positive electrode active material volume fraction": x0[1],
            }
        )
        experiment = pybamm.Experiment(
            [
                (
                    "Discharge at 1C for 3 minutes (1 second period)",
                    "Rest for 2 minutes (1 second period)",
                    "Charge at 1C for 3 minutes (1 second period)",
                    "Rest for 2 minutes (1 second period)",
                ),
            ]
            * 2
        )
        sim = model.predict(init_soc=init_soc, experiment=experiment)
        return sim
