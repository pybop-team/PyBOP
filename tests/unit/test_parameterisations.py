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

        # Form dataset
        x0 = np.array([0.52, 0.63])
        solution = self.getdata(model, x0)

        dataset = [
            pybop.Dataset("Time [s]", solution["Time [s]"].data),
            pybop.Dataset("Current function [A]", solution["Current [A]"].data),
            pybop.Dataset("Voltage [V]", solution["Terminal voltage [V]"].data),
        ]

        # Fitting parameters
        parameters = [
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

        # Define the cost to optimise
        cost = pybop.RMSE()
        signal = "Voltage [V]"

        # Select optimiser
        optimiser = pybop.NLoptOptimize(n_param=len(parameters))

        # Build the optimisation problem
        parameterisation = pybop.Optimisation(
            cost=cost,
            model=model,
            optimiser=optimiser,
            parameters=parameters,
            dataset=dataset,
            signal=signal,
        )

        # Run the optimisation problem
        x, _, final_cost, _ = parameterisation.run()

        # Assertions (for testing purposes only)
        np.testing.assert_allclose(final_cost, 0, atol=1e-2)
        np.testing.assert_allclose(x, x0, atol=1e-1)

    @pytest.mark.unit
    def test_spme_multiple_optimisers(self):
        # Define model
        model = pybop.lithium_ion.SPMe()
        model.parameter_set = model.pybamm_model.default_parameter_values

        # Form dataset
        x0 = np.array([0.52, 0.63])
        solution = self.getdata(model, x0)

        dataset = [
            pybop.Dataset("Time [s]", solution["Time [s]"].data),
            pybop.Dataset("Current function [A]", solution["Current [A]"].data),
            pybop.Dataset("Voltage [V]", solution["Terminal voltage [V]"].data),
        ]

        # Fitting parameters
        parameters = [
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

        # Define the cost to optimise
        cost = pybop.RMSE()
        signal = "Voltage [V]"

        # Select optimisers
        optimisers = [
            pybop.NLoptOptimize(n_param=len(parameters)),
            pybop.SciPyMinimize()
        ]

        # Test each optimiser
        for optimiser in optimisers:

            parameterisation = pybop.Optimisation(
                cost=cost,
                model=model,
                optimiser=optimiser,
                parameters=parameters,
                dataset=dataset,
                signal=signal,
            )

            # Run the optimisation problem
            x, _, final_cost, _ = parameterisation.run()
            # Assertions (for testing purposes only)
            np.testing.assert_allclose(final_cost, 0, atol=1e-2)
            np.testing.assert_allclose(x, x0, rtol=1e-1)

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


if __name__ == "__main__":
    unittest.main()
