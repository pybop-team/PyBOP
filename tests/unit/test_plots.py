import numpy as np
import pytest

import pybop


class TestPlots:
    """
    A class to test the plotting classes.
    """

    @pytest.fixture
    def model(self):
        # Define an example model
        return pybop.lithium_ion.SPM()

    @pytest.mark.unit
    def test_model_plots(self):
        # Test plotting of Model objects
        pass

    @pytest.fixture
    def problem(self, model):
        # Define an example problem
        parameters = [
            pybop.Parameter(
                "Negative particle radius [m]",
                prior=pybop.Gaussian(6e-06, 0.1e-6),
                bounds=[1e-6, 9e-6],
            ),
            pybop.Parameter(
                "Positive particle radius [m]",
                prior=pybop.Gaussian(4.5e-06, 0.1e-6),
                bounds=[1e-6, 9e-6],
            ),
        ]

        # Generate data
        t_eval = np.arange(0, 50, 2)
        values = model.predict(t_eval=t_eval)

        # Form dataset
        dataset = pybop.Dataset(
            {
                "Time [s]": t_eval,
                "Current function [A]": values["Current [A]"].data,
                "Voltage [V]": values["Voltage [V]"].data,
            }
        )

        # Generate problem
        return pybop.FittingProblem(model, parameters, dataset)

    @pytest.mark.unit
    def test_problem_plots(self):
        # Test plotting of Problem objects
        pass

    @pytest.fixture
    def cost(self, problem):
        # Define an example cost
        return pybop.SumSquaredError(problem)

    @pytest.mark.unit
    def test_cost_plots(self, cost):
        # Test plotting of Cost objects
        pybop.quick_plot(cost.x0, cost, title="Optimised Comparison")

        # Plot the cost landscape
        pybop.plot_cost2d(cost, steps=5)

        # Test without bounds
        for param in cost.problem.parameters:
            param.bounds = None
        with pytest.raises(ValueError):
            pybop.plot_cost2d(cost, steps=5)
        pybop.plot_cost2d(cost, bounds=np.array([[1e-6, 9e-6], [1e-6, 9e-6]]), steps=5)

    @pytest.fixture
    def optim(self, cost):
        # Define and run an example optimisation
        optim = pybop.Optimisation(cost, optimiser=pybop.CMAES)
        optim.run()
        return optim

    @pytest.mark.unit
    def test_optim_plots(self, optim):
        # Plot convergence
        pybop.plot_convergence(optim)

        # Plot the parameter traces
        pybop.plot_parameters(optim)

        # Plot the cost landscape with optimisation path
        pybop.plot_cost2d(optim.cost, optim=optim, steps=5)
