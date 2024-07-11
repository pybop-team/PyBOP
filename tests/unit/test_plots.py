import warnings

import numpy as np
import pytest
from packaging import version

import pybop


class TestPlots:
    """
    A class to test the plotting classes.
    """

    @pytest.mark.unit
    def test_standard_plot(self):
        # Test standard plot
        trace_names = pybop.StandardPlot.remove_brackets(["Trace [1]", "Trace [2]"])
        plot_dict = pybop.StandardPlot(
            x=np.ones((2, 10)),
            y=np.ones((2, 10)),
            trace_names=trace_names,
        )
        plot_dict()

    @pytest.fixture
    def model(self):
        # Define an example model
        return pybop.lithium_ion.SPM()

    @pytest.fixture
    def parameters(self):
        return pybop.Parameters(
            pybop.Parameter(
                "Negative electrode active material volume fraction",
                prior=pybop.Gaussian(0.68, 0.05),
                bounds=[0.5, 0.8],
            ),
            pybop.Parameter(
                "Positive electrode active material volume fraction",
                prior=pybop.Gaussian(0.58, 0.05),
                bounds=[0.4, 0.7],
            ),
        )

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
        pybop.plot_trajectories(
            dataset["Time [s]"],
            dataset["Voltage [V]"],
            trace_names=["Time [s]", "Voltage [V]"],
        )
        pybop.plot_dataset(dataset)

    @pytest.fixture
    def fitting_problem(self, model, parameters, dataset):
        return pybop.FittingProblem(model, parameters, dataset)

    @pytest.fixture
    def experiment(self):
        return pybop.Experiment(
            [
                ("Discharge at 1C for 10 minutes (20 second period)"),
            ]
        )

    @pytest.fixture
    def design_problem(self, model, parameters, experiment):
        return pybop.DesignProblem(model, parameters, experiment)

    @pytest.mark.unit
    def test_problem_plots(self, fitting_problem, design_problem):
        # Test plotting of Problem objects
        pybop.quick_plot(fitting_problem, title="Optimised Comparison")
        pybop.quick_plot(design_problem)

        # Test conversion of values into inputs
        pybop.quick_plot(fitting_problem, problem_inputs=[0.6, 0.6])

    @pytest.fixture
    def cost(self, fitting_problem):
        # Define an example cost
        return pybop.SumSquaredError(fitting_problem)

    @pytest.mark.unit
    def test_cost_plots(self, cost):
        # Test plotting of Cost objects
        pybop.plot2d(cost, gradient=True, steps=5)

        # Test without bounds
        for param in cost.problem.parameters:
            param.bounds = None
        with pytest.raises(ValueError):
            pybop.plot2d(cost, steps=5)
        pybop.plot2d(cost, bounds=np.array([[0.5, 0.8], [0.4, 0.7]]), steps=5)

    @pytest.fixture
    def optim(self, cost):
        # Define and run an example optimisation
        optim = pybop.Optimisation(cost)
        optim.run()
        return optim

    @pytest.mark.unit
    def test_optim_plots(self, optim):
        # Plot convergence
        pybop.plot_convergence(optim)
        optim._minimising = False
        pybop.plot_convergence(optim)

        # Plot the parameter traces
        pybop.plot_parameters(optim)

        # Plot the cost landscape with optimisation path
        pybop.plot2d(optim, steps=5)

        # Plot the cost landscape using optimisation path
        pybop.plot2d(optim, steps=5, use_optim_log=True)

        # Plot gradient cost landscape
        pybop.plot2d(optim, gradient=True, steps=5)

    @pytest.mark.unit
    def test_with_ipykernel(self, dataset, cost, optim):
        import ipykernel

        assert version.parse(ipykernel.__version__) >= version.parse("0.6")
        pybop.plot_dataset(dataset, signal=["Voltage [V]"])
        pybop.plot2d(cost, gradient=True, steps=5)
        pybop.plot_convergence(optim)
        pybop.plot_parameters(optim)
        pybop.plot2d(optim, steps=5)

    @pytest.mark.unit
    def test_gaussianlogliklihood_plots(self, fitting_problem):
        # Test plotting of GaussianLogLikelihood
        likelihood = pybop.GaussianLogLikelihood(fitting_problem)
        optim = pybop.CMAES(likelihood, max_iterations=5)
        optim.run()

        # Plot parameters
        pybop.plot_parameters(optim)

    @pytest.mark.unit
    def test_plot2d_incorrect_number_of_parameters(self, model, dataset):
        # Test with less than two paramters
        parameters = pybop.Parameters(
            pybop.Parameter(
                "Negative electrode active material volume fraction",
                prior=pybop.Gaussian(0.68, 0.05),
                bounds=[0.5, 0.8],
            ),
        )
        fitting_problem = pybop.FittingProblem(model, parameters, dataset)
        cost = pybop.SumSquaredError(fitting_problem)
        with pytest.raises(
            ValueError, match="This cost function takes fewer than 2 parameters."
        ):
            pybop.plot2d(cost)

        # Test with more than two paramters
        parameters = pybop.Parameters(
            pybop.Parameter(
                "Negative electrode active material volume fraction",
                prior=pybop.Gaussian(0.68, 0.05),
                bounds=[0.5, 0.8],
            ),
            pybop.Parameter(
                "Positive electrode active material volume fraction",
                prior=pybop.Gaussian(0.58, 0.05),
                bounds=[0.4, 0.7],
            ),
            pybop.Parameter(
                "Positive particle radius [m]",
                prior=pybop.Gaussian(4.8e-06, 0.05e-06),
                bounds=[4e-06, 6e-06],
            ),
        )
        fitting_problem = pybop.FittingProblem(model, parameters, dataset)
        cost = pybop.SumSquaredError(fitting_problem)
        pybop.plot2d(cost)

    @pytest.mark.unit
    def test_plot2d_prior_bounds(self, model, dataset):
        # Test with prior bounds
        parameters = pybop.Parameters(
            pybop.Parameter(
                "Negative electrode active material volume fraction",
                prior=pybop.Gaussian(0.68, 0.01),
            ),
            pybop.Parameter(
                "Positive electrode active material volume fraction",
                prior=pybop.Gaussian(0.58, 0.01),
            ),
        )
        fitting_problem = pybop.FittingProblem(model, parameters, dataset)
        cost = pybop.SumSquaredError(fitting_problem)
        with pytest.warns(
            UserWarning,
            match="Bounds were created from prior distributions.",
        ):
            warnings.simplefilter("always")
            pybop.plot2d(cost)
