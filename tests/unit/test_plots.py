import pybop
import numpy as np
import pytest


class TestPlots:
    """
    A class to test the plotting classes.
    """

    @pytest.fixture
    def model(self):
        # Define an example model
        return pybop.lithium_ion.SPM()

    @pytest.fixture
    def parameters(self):
        return [
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

    @pytest.fixture
    def dataset(self, model):
        # Generate data
        t_eval = np.arange(0, 50, 2)
        values = model.predict(t_eval=t_eval)

        # Form dataset
        return pybop.Dataset(
            {
                "Time [s]": t_eval,
                "Current function [A]": values["Current [A]"].data,
                "Voltage [V]": values["Voltage [V]"].data,
            }
        )

    @pytest.mark.unit
    def test_dataset_plots(self, dataset):
        # Test plotting of Dataset objects
        pybop.plot_dataset(dataset, signal=["Voltage [V]"])

    @pytest.fixture
    def problem(self, model, parameters, dataset):
        return pybop.FittingProblem(model, parameters, dataset)

    @pytest.mark.unit
    def test_problem_plots(self, problem):
        # Test plotting of Problem objects
        pybop.quick_plot(problem, title="Optimised Comparison")

    @pytest.fixture
    def cost(self, problem):
        # Define an example cost
        return pybop.SumSquaredError(problem)

    @pytest.mark.unit
    def test_cost_plots(self, cost):
        # Test plotting of Cost objects
        pybop.plot_cost2d(cost, steps=5)

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
        pybop.plot_optim2d(optim, steps=5)
