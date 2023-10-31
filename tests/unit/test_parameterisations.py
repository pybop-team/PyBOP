import pybop
import pybamm
import pytest
import numpy as np
import unittest


class TestModelParameterisation(unittest.TestCase):
    """
    A class to test the model parameterisation methods.
    """

    @pytest.mark.unit
    def test_spm(self):
        # Define model
        model = pybop.lithium_ion.SPM()
        model.parameter_set = model.pybamm_model.default_parameter_values

        # Form observations
        x0 = np.array([0.52, 0.63])
        solution = self.getdata(model, x0)

        observations = [
            pybop.Dataset("Time [s]", solution["Time [s]"].data),
            pybop.Dataset("Current function [A]", solution["Current [A]"].data),
            pybop.Dataset("Voltage [V]", solution["Terminal voltage [V]"].data),
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
            model, observations=observations, fit_parameters=params
        )

        # get RMSE estimate using NLOpt
        results, last_optim, num_evals = parameterisation.rmse(
            signal="Voltage [V]", method="nlopt"
        )

        # Assertions (for testing purposes only)
        np.testing.assert_allclose(last_optim, 0, atol=1e-2)
        np.testing.assert_allclose(results, x0, atol=1e-1)

    @pytest.mark.unit
    def test_spme(self):
        # Define model
        model = pybop.lithium_ion.SPMe()
        model.parameter_set = model.pybamm_model.default_parameter_values

        # Form observations
        x0 = np.array([0.52, 0.63])
        solution = self.getdata(model, x0)

        observations = [
            pybop.Dataset("Time [s]", solution["Time [s]"].data),
            pybop.Dataset("Current function [A]", solution["Current [A]"].data),
            pybop.Dataset("Voltage [V]", solution["Terminal voltage [V]"].data),
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
            model, observations=observations, fit_parameters=params
        )

        # get RMSE estimate using NLOpt
        results, last_optim, num_evals = parameterisation.rmse(
            signal="Voltage [V]", method="nlopt"
        )

        # Assertions (for testing purposes only)
        np.testing.assert_allclose(last_optim, 0, atol=1e-2)
        np.testing.assert_allclose(results, x0, rtol=1e-1)

    def getdata(self, model, x0):
        model.parameter_set = model.pybamm_model.default_parameter_values

        model.parameter_set.update(
            {
                "Negative electrode active material volume fraction": x0[0],
                "Positive electrode active material volume fraction": x0[1],
            }
        )
        experiment = pybamm.Experiment(
            [
                (
                    "Discharge at 2C for 5 minutes (1 second period)",
                    "Rest for 2 minutes (1 second period)",
                    "Charge at 1C for 5 minutes (1 second period)",
                    "Rest for 2 minutes (1 second period)",
                ),
            ]
            * 2
        )
        sim = model.predict(experiment=experiment)
        return sim

    @pytest.mark.unit
    def test_simulate_without_build_model(self):
        # Define model
        model = pybop.lithium_ion.SPM()

        with pytest.raises(
            ValueError, match="Model must be built before calling simulate"
        ):
            model.simulate(None, None)

    @pytest.mark.unit
    def test_priors(self):
        # Tests priors
        Gaussian = pybop.Gaussian(0.5, 1)
        Uniform = pybop.Uniform(0, 1)
        Exponential = pybop.Exponential(1)

        np.testing.assert_allclose(Gaussian.pdf(0.5), 0.3989422804014327, atol=1e-4)
        np.testing.assert_allclose(Uniform.pdf(0.5), 1, atol=1e-4)
        np.testing.assert_allclose(Exponential.pdf(1), 0.36787944117144233, atol=1e-4)

    @pytest.mark.unit
    def test_parameter_set(self):
        # Tests parameter set creation
        with pytest.raises(ValueError):
            pybop.ParameterSet("pybamms", "Chen2020")

        parameter_test = pybop.ParameterSet("pybamm", "Chen2020")
        np.testing.assert_allclose(
            parameter_test["Negative electrode active material volume fraction"], 0.75
        )

if __name__ == "__main__":
    unittest.main()
