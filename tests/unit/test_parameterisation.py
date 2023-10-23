import unittest
import pybop
import pybamm
import numpy as np


class TestParameterisation(unittest.TestCase):
    """
    Tests the parameterisation functionality of PyBOP.
    """

    def getdata(self, model, x0):

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
        prediction = model.simulate(experiment=experiment)
        return prediction

    def test_spm_nlopt(self):
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

        # Define fitting parameters
        parameters = [
            pybop.Parameter(
                name="Negative electrode active material volume fraction",
                prior=pybop.Gaussian(0.5, 0.02),
                bounds=[0.375, 0.625],
            ),
            pybop.Parameter(
                name="Positive electrode active material volume fraction",
                prior=pybop.Gaussian(0.65, 0.02),
                bounds=[0.525, 0.75],
            ),
        ]

        # Define the cost to optimise
        cost = pybop.RMSE()
        signal = "Voltage [V]"

        # Select optimiser
        optimiser = pybop.NLoptOptimize(x0=x0)

        # Build the optimisation problem
        parameterisation = pybop.Optimisation(
            cost=cost, model=model,optimiser=optimiser,
            parameters=parameters, dataset=dataset, signal=signal,
        )

        # Run the optimisation problem
        x, output, final_cost, num_evals = parameterisation.run()

        # Check assertions
        np.testing.assert_allclose(final_cost, 1e-3, atol=1e-2)
        np.testing.assert_allclose(x, x0, atol=1e-1)

    def test_spme_scipy(self):
        # Define model
        model = pybop.lithium_ion.SPMe()
        model.parameter_set = model.pybamm_model.default_parameter_values

        # Form dataset
        x0 = np.array([0.52, 0.63])
        solution = self.getdata(model,x0)
        dataset = [
            pybop.Dataset("Time [s]", solution["Time [s]"].data),
            pybop.Dataset("Current function [A]", solution["Current [A]"].data),
            pybop.Dataset("Voltage [V]", solution["Terminal voltage [V]"].data),
        ]

        # Define fitting parameters
        parameters = [
            pybop.Parameter(
                name="Negative electrode active material volume fraction",
                prior=pybop.Gaussian(0.5, 0.02),
                bounds=[0.375, 0.625],
            ),
            pybop.Parameter(
                name="Positive electrode active material volume fraction",
                prior=pybop.Gaussian(0.65, 0.02),
                bounds=[0.525, 0.75],
            ),
        ]

        # Define the cost to optimise
        cost = pybop.RMSE()
        signal = "Voltage [V]"

        # Select optimiser
        optimiser = pybop.SciPyMinimize(x0=x0)

        # Build the optimisation problem
        parameterisation = pybop.Optimisation(
            cost=cost, model=model, optimiser=optimiser,
            parameters=parameters, dataset=dataset, signal=signal,
        )

        # Run the optimisation problem
        x, output, final_cost, num_evals = parameterisation.run()

        # Check assertions
        np.testing.assert_allclose(final_cost, 1e-3, atol=1e-2)
        np.testing.assert_allclose(x, x0, atol=1e-1)

    def test_spm_pints(self):
        # Define model
        model = pybop.lithium_ion.SPMe()
        model.parameter_set = model.pybamm_model.default_parameter_values

        # Form dataset
        x0 = np.array([0.52, 0.63])
        solution = self.getdata(model,x0)
        dataset = [
            pybop.Dataset("Time [s]", solution["Time [s]"].data),
            pybop.Dataset("Current function [A]", solution["Current [A]"].data),
            pybop.Dataset("Voltage [V]", solution["Terminal voltage [V]"].data),
        ]

        # Define fitting parameters
        parameters = [
            pybop.Parameter(
                name="Negative electrode active material volume fraction",
                prior=pybop.Gaussian(0.5, 0.02),
                bounds=[0.375, 0.625],
            ),
            pybop.Parameter(
                name="Positive electrode active material volume fraction",
                prior=pybop.Gaussian(0.65, 0.02),
                bounds=[0.525, 0.75],
            ),
        ]

        # Define the cost to optimise
        cost = pybop.RMSE()
        signal = "Voltage [V]"

        # Select optimiser
        optimiser = pybop.PintsOptimiser(x0=x0)

        # Build the optimisation problem
        parameterisation = pybop.Optimisation(
            cost=cost, model=model, optimiser=optimiser,
            parameters=parameters, dataset=dataset, signal=signal,
        )

        # Run the optimisation problem
        x, output, final_cost, num_evals = parameterisation.run()

        # Check assertions
        np.testing.assert_allclose(final_cost, 1e-3, atol=1e-2)
        np.testing.assert_allclose(x, x0, atol=1e-1)


if __name__ == '__main__':
    unittest.main()
