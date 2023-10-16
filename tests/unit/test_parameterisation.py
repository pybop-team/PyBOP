import pybop
import pybamm
import numpy as np


class TestParameterisation:
    def getdata(self, model, x0):

        # Define default parameter set
        model.parameter_set = model.pybamm_model.default_parameter_values

        # Update fitting parameters
        model.parameter_set.update(
            {
                "Negative electrode active material volume fraction": x0[0],
                "Positive electrode active material volume fraction": x0[1],
            }
        )

        # Define experimental protocol
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

        # Simulate model to generate test dataset
        sim = model.simulate(experiment=experiment)
        return sim

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

        # Define fitting parameters
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

        # Define optimisation problem
        parameterisation = pybop.Optimisation(
            model, observations=observations, fit_parameters=params
        )

        # Optimise RMSE using NLOpt
        results, last_optim, num_evals = parameterisation.rmse(
            signal="Voltage [V]", method="nlopt"
        )

        # Check assertions
        np.testing.assert_allclose(last_optim, 1e-3, atol=1e-2)
        np.testing.assert_allclose(results, x0, atol=1e-1)

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

        # Define fitting parameters
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

        # Define optimisation problem
        parameterisation = pybop.Optimisation(
            model, observations=observations, fit_parameters=params
        )

        # Optimise RMSE using NLOpt
        results, last_optim, num_evals = parameterisation.rmse(
            signal="Voltage [V]", method="nlopt"
        )

        # Check assertions
        np.testing.assert_allclose(last_optim, 1e-3, atol=1e-2)
        np.testing.assert_allclose(results, x0, atol=1e-1)
