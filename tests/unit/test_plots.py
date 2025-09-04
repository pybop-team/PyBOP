import numpy as np
import pybamm
import pytest
from packaging import version

import pybop
import pybop.builders


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
        return pybamm.lithium_ion.SPM()

    @pytest.fixture
    def parameters(self):
        return [
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
        ]

    @pytest.fixture
    def dataset(self, model):
        t_eval = np.arange(0, 50, 2)
        # Generate data
        values = pybamm.Simulation(model).solve(t_eval=[0.0, 50.0], t_interp=t_eval)

        # Form dataset
        return pybop.Dataset(
            {
                "Time [s]": values.t,
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
    def problem(self, model, parameters, dataset):
        builder = pybop.Pybamm()
        builder.set_simulation(model)
        builder.set_dataset(dataset)
        for p in parameters:
            builder.add_parameter(p)
        builder.add_cost(
            pybop.costs.pybamm.SumSquaredError("Voltage [V]", "Voltage [V]")
        )
        return builder.build()

    @pytest.fixture
    def likelihood_problem(self, model, parameters, dataset):
        builder = pybop.Pybamm()
        builder.set_simulation(model)
        builder.set_dataset(dataset)
        for p in parameters:
            builder.add_parameter(p)
        builder.add_cost(
            pybop.costs.pybamm.NegativeGaussianLogLikelihood(
                "Voltage [V]", "Voltage [V]", sigma=1e-3
            )
        )
        return builder.build()

    def test_cost_plots(self, problem):
        # Test plot of Cost objects
        pybop.plot.contour(problem, gradient=True, steps=5)
        pybop.plot.contour(problem, gradient=True, steps=5, apply_transform=True)

        # Test with bounds
        pybop.plot.contour(problem, bounds=np.array([[0.5, 0.8], [0.4, 0.7]]), steps=5)

    @pytest.fixture
    def result(self, problem):
        # Define and run an example optimisation
        options = pybop.XNES.default_options()
        options.max_iterations = 20
        optim = pybop.XNES(problem, options)
        return optim.run()

    def test_optim_plots(self, result):
        bounds = np.asarray([[0.5, 0.8], [0.4, 0.7]])

        # Plot convergence
        result.plot_convergence()

        # Plot the parameter traces
        result.plot_parameters()

        # Plot the cost landscape with optimisation path
        result.plot_contour(steps=3)

        # Plot the cost landscape with optim trace and bounds
        result.plot_contour(steps=3, bounds=bounds)

        # Plot the cost landscape using optimisation path
        result.plot_contour(steps=3, use_optim_log=True)

        # Plot gradient cost landscape
        result.plot_contour(gradient=True, steps=5)

        # Plot voronoi
        result.plot_surface(normalise=False)

        # Plot voronoi w/ bounds
        result.plot_surface(bounds=bounds)

        with pytest.raises(
            ValueError, match="Lower bounds must be strictly less than upper bounds."
        ):
            result.plot_surface(bounds=[[0.5, 0.8], [0.7, 0.4]])

    @pytest.fixture
    def posterior_summary(self, likelihood_problem):
        options = pybop.SliceStepoutMCMC.default_options()
        options.max_iterations = 1
        options.n_chains = 1
        sampler = pybop.SliceStepoutMCMC(likelihood_problem, options)
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

    def test_with_ipykernel(self, dataset, problem, result):
        import ipykernel

        assert version.parse(ipykernel.__version__) >= version.parse("0.6")
        pybop.plot.dataset(dataset, signal=["Voltage [V]"])
        pybop.plot.contour(problem, gradient=True, steps=5)
        pybop.plot.convergence(result)
        pybop.plot.parameters(result)
        pybop.plot.contour(result, steps=5)

    def test_gaussianloglikelihood_plots(self, likelihood_problem):
        options = pybop.CMAES.default_options()
        options.max_iterations = 5
        optim = pybop.CMAES(likelihood_problem, options)
        result = optim.run()

        # Plot parameters
        pybop.plot.parameters(result)

    def test_contour_incorrect_number_of_parameters(self, model, dataset):
        builder = pybop.Pybamm()
        builder.set_simulation(model)
        builder.set_dataset(dataset)
        builder.add_parameter(
            pybop.Parameter(
                "Negative electrode active material volume fraction",
                prior=pybop.Gaussian(0.68, 0.05),
                bounds=[0.5, 0.8],
            )
        )
        builder.add_cost(
            pybop.costs.pybamm.SumSquaredError("Voltage [V]", "Voltage [V]")
        )
        fitting_problem = builder.build()
        with pytest.raises(
            ValueError, match="This problem takes fewer than 2 parameters."
        ):
            pybop.plot.contour(fitting_problem, steps=5)

        # Test with more than two paramters
        builder.add_parameter(
            pybop.Parameter(
                "Positive electrode active material volume fraction",
                prior=pybop.Gaussian(0.58, 0.05),
                bounds=[0.4, 0.7],
            )
        )
        builder.add_parameter(
            pybop.Parameter(
                "Positive particle radius [m]",
                prior=pybop.Gaussian(4.8e-06, 0.05e-06),
                bounds=[4e-06, 6e-06],
            )
        )
        fitting_problem = builder.build()
        pybop.plot.contour(fitting_problem, steps=5)

    def test_nyquist(self):
        # Define model
        model = pybamm.lithium_ion.SPM(options={"surface form": "differential"})

        # Fitting parameters
        parameter = pybop.Parameter(
            "Positive electrode thickness [m]",
            prior=pybop.Gaussian(60e-6, 1e-6),
            bounds=[10e-6, 80e-6],
        )

        # Form dataset
        dataset = pybop.Dataset(
            {
                "Frequency [Hz]": np.logspace(-4, 5, 10),
                "Current function [A]": np.ones(10) * 0.0,
                "Impedance": np.ones(10) * 0.0,
            }
        )
        builder = pybop.builders.PybammEIS()
        builder.set_simulation(model)
        builder.set_dataset(dataset)
        builder.add_parameter(parameter)
        builder.add_cost(pybop.costs.SumSquaredError())
        problem = builder.build()

        # Plot the nyquist
        pybop.plot.nyquist(
            problem,
            problem_inputs={"Positive electrode thickness [m]": 60e-6},
            title="Optimised Comparison",
        )

        # Without inputs
        pybop.plot.nyquist(problem, title="Optimised Comparison")

    def test_validation_plot(self, problem, dataset):
        pybop.plot.validation(
            problem.params.get_values(), problem=problem, dataset=dataset
        )
