import warnings

import numpy as np
import pybamm
import pytest
from packaging import version

import pybop


class TestPlots:
    """
    A class to test the plot classes.
    """

    pytestmark = pytest.mark.unit

    def test_standard_plot(self):
        # Test standard plot
        trace_names = pybop.plot.StandardPlot.remove_brackets(
            ["Trace [1]", "Trace [2]"]
        )
        plot_dict = pybop.plot.StandardPlot(
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
                transformation=pybop.ScaledTransformation(
                    coefficient=1 / 0.3, intercept=-0.5
                ),
            ),
            pybop.Parameter(
                "Positive electrode active material volume fraction",
                prior=pybop.Gaussian(0.58, 0.05),
                bounds=[0.4, 0.7],
                transformation=pybop.ScaledTransformation(
                    coefficient=1 / 0.3, intercept=-0.4
                ),
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

    def test_dataset_plots(self, dataset):
        # Test plot of Dataset objects
        pybop.plot.trajectories(
            dataset["Time [s]"],
            dataset["Voltage [V]"],
            trace_names=["Time [s]", "Voltage [V]"],
        )
        pybop.plot.dataset(dataset)

    @pytest.fixture
    def fitting_problem(self, model, parameters, dataset):
        return pybop.FittingProblem(model, parameters, dataset)

    @pytest.fixture
    def jax_fitting_problem(self, model, parameters, dataset):
        problem = pybop.FittingProblem(model, parameters, dataset)
        problem.model.jaxify_solver(t_eval=problem.domain_data)
        return problem

    @pytest.fixture
    def experiment(self):
        return pybamm.Experiment(
            [
                ("Discharge at 1C for 10 minutes (20 second period)"),
            ]
        )

    @pytest.fixture
    def design_problem(self, model, parameters, experiment):
        model = pybop.lithium_ion.SPM()
        return pybop.DesignProblem(model, parameters, experiment)

    def test_problem_plots(self, fitting_problem, design_problem, jax_fitting_problem):
        # Test plot of Problem objects
        pybop.plot.problem(fitting_problem, title="Optimised Comparison")
        pybop.plot.problem(design_problem)
        pybop.plot.problem(jax_fitting_problem)

        # Test conversion of values into inputs
        pybop.plot.problem(fitting_problem, problem_inputs=[0.6, 0.6])

    @pytest.fixture
    def cost(self, fitting_problem):
        # Define an example cost
        return pybop.SumSquaredError(fitting_problem)

    def test_cost_plots(self, cost):
        # Test plot of Cost objects
        pybop.plot.contour(cost, gradient=True, steps=5)

        pybop.plot.contour(cost, gradient=True, steps=5, apply_transform=True)

        # Test without bounds
        for param in cost.parameters:
            param.bounds = None
        with pytest.raises(ValueError, match="All parameters require bounds for plot."):
            pybop.plot.contour(cost, steps=5)

        # Test with bounds
        pybop.plot.contour(cost, bounds=np.array([[0.5, 0.8], [0.4, 0.7]]), steps=5)

    @pytest.fixture
    def optim(self, cost):
        # Define and run an example optimisation
        optim = pybop.Optimisation(cost)
        optim.run()
        return optim

    def test_optim_plots(self, optim):
        bounds = np.asarray([[0.5, 0.8], [0.4, 0.7]])

        # Plot convergence
        pybop.plot.convergence(optim)
        optim.invert_cost = True
        pybop.plot.convergence(optim)

        # Plot the parameter traces
        pybop.plot.parameters(optim)

        # Plot the cost landscape with optimisation path
        pybop.plot.contour(optim, steps=3)

        # Plot the cost landscape w/ optim & bounds
        pybop.plot.contour(optim, steps=3, bounds=bounds)

        # Plot the cost landscape using optimisation path
        pybop.plot.contour(optim, steps=3, use_optim_log=True)

        # Plot gradient cost landscape
        pybop.plot.contour(optim, gradient=True, steps=5)

        # Plot voronoi
        pybop.plot.surface(optim, normalise=False)

        # Plot voronoi w/ bounds
        pybop.plot.surface(optim, bounds=bounds)

        with pytest.raises(
            ValueError, match="Lower bounds must be strictly less than upper bounds."
        ):
            pybop.plot.surface(optim, bounds=[[0.5, 0.8], [0.7, 0.4]])

    @pytest.fixture
    def posterior_summary(self, fitting_problem):
        posterior = pybop.LogPosterior(
            pybop.GaussianLogLikelihoodKnownSigma(fitting_problem, sigma0=2e-3)
        )
        sampler = pybop.SliceStepoutMCMC(posterior, chains=1, iterations=1)
        results = sampler.run()
        return pybop.PosteriorSummary(results)

    def test_posterior_plots(self, posterior_summary):
        # Plot trace
        posterior_summary.plot_trace()

        # Plot posterior
        posterior_summary.plot_posterior()

        # Plot chains
        posterior_summary.plot_chains()

        # Plot summary table
        posterior_summary.summary_table()

    def test_with_ipykernel(self, dataset, cost, optim):
        import ipykernel

        assert version.parse(ipykernel.__version__) >= version.parse("0.6")
        pybop.plot.dataset(dataset, signal=["Voltage [V]"])
        pybop.plot.contour(cost, gradient=True, steps=5)
        pybop.plot.convergence(optim)
        pybop.plot.parameters(optim)
        pybop.plot.contour(optim, steps=5)

    def test_gaussianloglikelihood_plots(self, fitting_problem):
        # Test plot of GaussianLogLikelihood
        likelihood = pybop.GaussianLogLikelihood(fitting_problem)
        optim = pybop.CMAES(likelihood, max_iterations=5)
        optim.run()

        # Plot parameters
        pybop.plot.parameters(optim)

    def test_contour_incorrect_number_of_parameters(self, model, dataset):
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
            pybop.plot.contour(cost)

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
        pybop.plot.contour(cost)

    def test_contour_prior_bounds(self, model, dataset):
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
            pybop.plot.contour(cost)

    def test_nyquist(self):
        # Define model
        model = pybop.lithium_ion.SPM(
            eis=True, options={"surface form": "differential"}
        )

        # Fitting parameters
        parameters = pybop.Parameters(
            pybop.Parameter(
                "Positive electrode thickness [m]",
                prior=pybop.Gaussian(60e-6, 1e-6),
                bounds=[10e-6, 80e-6],
            )
        )

        # Form dataset
        dataset = pybop.Dataset(
            {
                "Frequency [Hz]": np.logspace(-4, 5, 10),
                "Current function [A]": np.ones(10) * 0.0,
                "Impedance": np.ones(10) * 0.0,
            }
        )

        signal = ["Impedance"]
        # Generate problem, cost function, and optimisation class
        problem = pybop.FittingProblem(model, parameters, dataset, signal=signal)

        # Plot the nyquist
        pybop.plot.nyquist(
            problem, problem_inputs=[60e-6], title="Optimised Comparison"
        )

        # Without inputs
        pybop.plot.nyquist(problem, title="Optimised Comparison")
