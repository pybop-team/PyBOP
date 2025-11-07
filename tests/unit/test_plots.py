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
        return pybamm.lithium_ion.SPM()

    @pytest.fixture
    def parameters(self):
        return {
            "Negative electrode active material volume fraction": pybop.Parameter(
                prior=pybop.Gaussian(0.68, 0.05),
                bounds=[0.5, 0.8],
                transformation=pybop.ScaledTransformation(
                    coefficient=1 / 0.3, intercept=-0.5
                ),
            ),
            "Positive electrode active material volume fraction": pybop.Parameter(
                prior=pybop.Gaussian(0.58, 0.05),
                bounds=[0.4, 0.7],
                transformation=pybop.ScaledTransformation(
                    coefficient=1 / 0.3, intercept=-0.4
                ),
            ),
        }

    @pytest.fixture
    def dataset(self, model):
        t_eval = np.arange(0, 50, 2)
        solution = pybamm.Simulation(model).solve(t_eval=t_eval)
        return pybop.Dataset(
            {
                "Time [s]": t_eval,
                "Current function [A]": solution["Current [A]"](t_eval),
                "Voltage [V]": solution["Voltage [V]"](t_eval),
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
        parameter_values = model.default_parameter_values
        parameter_values.update(parameters)
        simulator = pybop.pybamm.Simulator(
            model, parameter_values=parameter_values, protocol=dataset
        )
        cost = pybop.SumSquaredError(dataset)
        return pybop.Problem(simulator, cost)

    @pytest.fixture
    def experiment(self):
        return pybamm.Experiment(["Discharge at 1C for 10 minutes (20 second period)"])

    @pytest.fixture
    def design_problem(self, model, parameters, experiment):
        parameter_values = model.default_parameter_values
        pybop.pybamm.set_formation_concentrations(parameter_values)
        parameter_values.update(parameters)
        simulator = pybop.pybamm.Simulator(
            model, parameter_values=parameter_values, protocol=experiment
        )
        return pybop.Problem(simulator)

    def test_problem_plots(self, fitting_problem, design_problem):
        # Test plot of Problem objects
        pybop.plot.problem(fitting_problem, title="Optimised Comparison")
        pybop.plot.problem(design_problem)

        # Test conversion of values into inputs
        pybop.plot.problem(fitting_problem, inputs=fitting_problem.parameters.to_dict([0.6, 0.6]))

    def test_cost_plots(self, fitting_problem):
        # Test plot of Cost objects
        pybop.plot.contour(fitting_problem, gradient=True, steps=5)

        pybop.plot.contour(fitting_problem, gradient=True, steps=5, transformed=True)

        # Test without bounds
        fitting_problem.parameters.remove_bounds()
        with pytest.raises(ValueError, match="All parameters require bounds for plot."):
            pybop.plot.contour(fitting_problem, steps=5)

        # Test with bounds
        pybop.plot.contour(
            fitting_problem, bounds=np.array([[0.5, 0.8], [0.4, 0.7]]), steps=5
        )

    @pytest.fixture
    def result(self, fitting_problem):
        # Define and run an example optimisation
        optim = pybop.XNES(fitting_problem)
        return optim.run()

    def test_optim_plots(self, result):
        bounds = np.asarray([[0.5, 0.8], [0.4, 0.7]])

        # Plot convergence
        result.plot_convergence()

        # Plot the parameter traces
        result.plot_parameters()

        # Plot the cost landscape with optimisation path
        result.plot_contour(steps=3)

        # Plot the cost landscape w/ optim & bounds
        result.plot_contour(steps=3, bounds=bounds)

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
    def posterior_summary(self, model, parameters, dataset):
        parameter_values = model.default_parameter_values
        parameter_values.update(parameters)
        simulator = pybop.pybamm.Simulator(
            model, parameter_values=parameter_values, protocol=dataset
        )
        likelihood = pybop.GaussianLogLikelihoodKnownSigma(dataset, sigma0=2e-3)
        posterior = pybop.LogPosterior(likelihood)
        problem = pybop.Problem(simulator, posterior)
        options = pybop.PintsSamplerOptions(n_chains=1, max_iterations=1)
        sampler = pybop.SliceStepoutMCMC(problem, options=options)
        result = sampler.run()
        return pybop.PosteriorSummary(result.chains)

    def test_posterior_plots(self, posterior_summary):
        # Plot trace
        posterior_summary.plot_trace()

        # Plot posterior
        posterior_summary.plot_posterior()

        # Plot chains
        posterior_summary.plot_chains()

        # Plot summary table
        posterior_summary.summary_table()

    def test_with_ipykernel(self, dataset, fitting_problem, result):
        import ipykernel

        assert version.parse(ipykernel.__version__) >= version.parse("0.6")
        pybop.plot.dataset(dataset, signal=["Voltage [V]"])
        pybop.plot.contour(fitting_problem, gradient=True, steps=5)
        result.plot_convergence()
        result.plot_parameters()
        result.plot_contour(steps=5)

    def test_contour_incorrect_number_of_parameters(self, model, dataset):
        parameter_values = model.default_parameter_values

        # Test with less than two paramters
        parameter_values.update(
            {
                "Negative electrode active material volume fraction": pybop.Parameter(
                    prior=pybop.Gaussian(0.68, 0.05),
                    bounds=[0.5, 0.8],
                ),
            }
        )
        simulator = pybop.pybamm.Simulator(
            model, parameter_values=parameter_values, protocol=dataset
        )
        cost = pybop.SumSquaredError(dataset)
        fitting_problem = pybop.Problem(simulator, cost)
        with pytest.raises(
            ValueError, match="This cost function takes fewer than 2 parameters."
        ):
            pybop.plot.contour(fitting_problem)

        # Test with more than two paramters
        parameter_values.update(
            {
                "Positive electrode active material volume fraction": pybop.Parameter(
                    prior=pybop.Gaussian(0.58, 0.05),
                    bounds=[0.4, 0.7],
                ),
                "Positive particle radius [m]": pybop.Parameter(
                    prior=pybop.Gaussian(4.8e-06, 0.05e-06),
                    bounds=[4e-06, 6e-06],
                ),
            }
        )
        simulator = pybop.pybamm.Simulator(
            model, parameter_values=parameter_values, protocol=dataset
        )
        cost = pybop.SumSquaredError(dataset)
        fitting_problem = pybop.Problem(simulator, cost)
        pybop.plot.contour(fitting_problem)

    def test_nyquist(self):
        # Define model
        model = pybamm.lithium_ion.SPM(options={"surface form": "differential"})
        parameter_values = model.default_parameter_values

        # Fitting parameters
        parameter_values.update(
            {
                "Positive electrode thickness [m]": pybop.Parameter(
                    prior=pybop.Gaussian(60e-6, 1e-6),
                    bounds=[10e-6, 80e-6],
                )
            }
        )

        # Form dataset
        dataset = pybop.Dataset(
            {
                "Frequency [Hz]": np.logspace(-4, 5, 10),
                "Current function [A]": np.ones(10) * 0.0,
                "Impedance": np.ones(10) * 0.0,
            },
            domain="Frequency [Hz]",
        )

        # Generate problem, cost function, and optimisation class
        simulator = pybop.pybamm.EISSimulator(
            model,
            parameter_values=parameter_values,
            f_eval=dataset["Frequency [Hz]"],
        )
        cost = pybop.MeanAbsoluteError(dataset, target="Impedance")
        problem = pybop.Problem(simulator, cost)

        # Plot the nyquist
        inputs = problem.parameters.to_dict([60e-6])
        pybop.plot.nyquist(problem, inputs=inputs, title="Optimised Comparison")

        # Without inputs
        pybop.plot.nyquist(problem, title="Optimised Comparison")
