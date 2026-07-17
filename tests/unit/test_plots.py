import matplotlib.pyplot as plt
import numpy as np
import pybamm
import pytest
from packaging import version

import pybop


@pytest.mark.parametrize("backend", ["plotly", "matplotlib"])
class TestPlots:
    """
    A class to test the plot classes.
    """

    pytestmark = pytest.mark.unit

    @pytest.fixture
    def figure_input(self, backend):
        if backend.lower() == "matplotlib":
            return plt.figure(), plt.gca()
        elif backend.lower() == "plotly":
            fig = pybop.plot.backends.PlotlyManager().make_subplots(
                rows=2, cols=1, specs=[[{}], [{"type": "table"}]]
            )
            ax = (1, 1)
            return fig, ax

    @pytest.fixture
    def model(self):
        return pybamm.lithium_ion.SPM()

    @pytest.fixture
    def parameters(self):
        return {
            "Negative electrode active material volume fraction": pybop.Parameter(
                distribution=pybop.Gaussian(0.68, 0.05, truncated_at=[0.5, 0.8]),
                transformation=pybop.ScaledTransformation(
                    coefficient=1 / 0.3, intercept=-0.5
                ),
            ),
            "Positive electrode active material volume fraction": pybop.Parameter(
                distribution=pybop.Gaussian(0.58, 0.05, truncated_at=[0.4, 0.7]),
                transformation=pybop.ScaledTransformation(
                    coefficient=1 / 0.3, intercept=-0.4
                ),
            ),
        }

    @pytest.fixture
    def parameters_no_bounds(self):
        return {
            "Negative electrode active material volume fraction": pybop.Parameter(
                distribution=pybop.Gaussian(0.68, 0.05),
                transformation=pybop.ScaledTransformation(
                    coefficient=1 / 0.3, intercept=-0.5
                ),
            ),
            "Positive electrode active material volume fraction": pybop.Parameter(
                distribution=pybop.Gaussian(0.58, 0.05),
                transformation=pybop.ScaledTransformation(
                    coefficient=1 / 0.3, intercept=-0.4
                ),
            ),
        }

    @pytest.fixture
    def dataset(self, model):
        t_eval = np.arange(0, 50, 2)
        solution = pybamm.Simulation(model).solve(t_eval=t_eval, t_interp=t_eval)
        return pybop.import_pybamm_solution(solution)

    def test_dataset_plots(self, dataset, backend, figure_input):
        pybop.plot.use_backend(backend)
        fig, ax = figure_input
        # Test plot of Dataset objects
        pybop.plot.trajectories(
            dataset["Time [s]"],
            dataset["Voltage [V]"],
            labels=["Time [s]", "Voltage [V]"],
        )
        pybop.plot.dataset(dataset)

        fig = pybop.plot.dataset(
            dataset, signal=["Voltage [V]"], figures=fig, axes=[ax], show=False
        )
        assert fig is not None

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
    def fitting_problem_no_bounds(self, model, parameters_no_bounds, dataset):
        parameter_values = model.default_parameter_values
        parameter_values.update(parameters_no_bounds)
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

    def test_problem_plots(
        self, fitting_problem, design_problem, backend, figure_input
    ):
        pybop.plot.use_backend(backend)
        fig, ax = figure_input
        # Test plot of Problem objects
        pybop.plot.problem(
            fitting_problem, title="Optimised Comparison", figures=fig, axes=[ax]
        )
        fig = pybop.plot.problem(design_problem, show=False)
        assert fig is not None

        # Test conversion of values into inputs
        pybop.plot.problem(
            fitting_problem, inputs=fitting_problem.parameters.to_dict([0.6, 0.6])
        )

    def test_cost_plots(
        self, fitting_problem, fitting_problem_no_bounds, backend, figure_input
    ):
        pybop.plot.use_backend(backend)
        # Test plot of Cost objects
        fig, ax = figure_input

        pybop.plot.contour(
            fitting_problem, gradient=True, steps=5, figures=fig, axes=[ax] * 3
        )

        pybop.plot.contour(
            fitting_problem,
            gradient=True,
            steps=5,
            transformed=True,
            figures=[fig],
            axes=[ax] * 3,
        )

        # Test without bounds
        with pytest.raises(ValueError, match="All parameters require bounds for plot."):
            pybop.plot.contour(fitting_problem_no_bounds, steps=5)

        # Test with bounds and show=False
        fig = pybop.plot.contour(
            fitting_problem,
            bounds=np.array([[0.5, 0.8], [0.4, 0.7]]),
            steps=5,
            show=False,
        )
        assert fig is not None

    @pytest.fixture
    def result(self, fitting_problem):
        # Define and run an example optimisation
        optim = pybop.XNES(fitting_problem)
        return optim.run()

    def test_optim_plots(self, result, backend, figure_input):
        pybop.plot.use_backend(backend)
        bounds = np.asarray([[0.5, 0.8], [0.4, 0.7]])
        fig, ax = figure_input

        # Plot convergence
        result.plot_convergence(figures=fig, axes=[ax])

        # Plot the parameter traces
        fig2 = result.plot_parameters(show=False)
        assert fig2 is not None

        # Plot the cost landscape with optimisation path
        result.plot_contour(steps=3)

        # Plot the cost landscape w/ optim & bounds
        result.plot_contour(steps=3, bounds=bounds)

        # Plot gradient cost landscape
        fig2, grad_figs = result.plot_contour(gradient=True, steps=5, show=False)
        assert fig2 is not None
        assert len(grad_figs) == len(result.problem.parameters)

        # Plot voronoi
        fig2 = result.plot_surface(normalise=False, show=False)
        assert fig2 is not None

        # Plot voronoi w/ bounds
        result.plot_surface(bounds=bounds, figures=fig, axes=[ax])

        with pytest.raises(
            ValueError, match="Lower bounds must be strictly less than upper bounds."
        ):
            result.plot_surface(bounds=[[0.5, 0.8], [0.7, 0.4]])

        with pytest.raises(
            ValueError, match="This plot method requires two parameters."
        ):
            result._x_model = [np.ones((np.shape(result._x_model)[1], 1))]
            result.plot_surface()

    @pytest.fixture
    def sampling_result(self, model, parameters, dataset):
        parameter_values = model.default_parameter_values
        parameter_values.update(parameters)
        simulator = pybop.pybamm.Simulator(
            model, parameter_values=parameter_values, protocol=dataset
        )
        log_likelihood = pybop.GaussianLogLikelihood(dataset)
        log_pdf = pybop.LogPosterior(simulator, log_likelihood)
        options = pybop.PintsSamplerOptions(n_chains=1, max_iterations=2)
        sampler = pybop.SliceStepoutMCMC(log_pdf, options=options)
        return sampler.run()

    def test_posterior_plots(self, sampling_result, backend, figure_input):
        pybop.plot.use_backend(backend)
        fig, ax = figure_input
        sampling_result.get_summary_statistics()

        # Plot trace
        f = sampling_result.plot_trace(show=False)
        assert f is not None
        sampling_result.plot_trace(figures=fig, axes=[ax])

        # Plot posterior
        f = sampling_result.plot_posterior(show=False)
        assert f is not None
        sampling_result.plot_posterior(figures=fig, axes=[ax])

        # Plot chains
        f = sampling_result.plot_chains(show=False)
        assert f is not None
        sampling_result.plot_chains(figures=fig, axes=[ax])

        # Plot posterior predictions
        f = sampling_result.plot_predictive(show=False, pdf_plot=[[1, 2], [2, 3]])
        assert f is not None
        sampling_result.plot_predictive(figures=fig, axes=[ax])

        # Plot the prior and posterior distributions
        f = pybop.plot.distribution(
            sampling_result.problem.parameters, sampling_result.posterior, show=False
        )
        assert f is not None
        pybop.plot.distribution(
            sampling_result.problem.parameters,
            sampling_result.posterior,
            figures=fig,
            axes=[ax],
        )

        # Plot summary table
        f = sampling_result.summary_table(show=False)
        assert f is not None
        if backend.lower() == "plotly":
            ax = (2, 1)  # Need correct plot type for table
        sampling_result.summary_table(figures=fig, axes=[ax])

    def test_with_ipykernel(self, dataset, fitting_problem, result, backend):
        import ipykernel

        pybop.plot.use_backend(backend)

        assert version.parse(ipykernel.__version__) >= version.parse("0.6")
        pybop.plot.dataset(dataset, signal=["Voltage [V]"])
        pybop.plot.contour(fitting_problem, gradient=True, steps=5)
        fig = result.plot_convergence(show=False)
        assert fig is not None
        backend = pybop.plot.get_backend(backend)
        backend.show_figure(fig)
        result.plot_convergence()
        result.plot_parameters()
        result.plot_contour(steps=5)

    def test_contour_incorrect_number_of_parameters(self, model, dataset, backend):
        pybop.plot.use_backend(backend)
        parameter_values = model.default_parameter_values

        # Test with less than two paramters
        parameter_values.update(
            {
                "Negative electrode active material volume fraction": pybop.Parameter(
                    distribution=pybop.Gaussian(0.68, 0.05, truncated_at=[0.5, 0.8]),
                )
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
                    pybop.Gaussian(0.58, 0.05, truncated_at=[0.4, 0.7])
                ),
                "Positive particle radius [m]": pybop.Parameter(
                    distribution=pybop.Gaussian(
                        4.8e-06, 0.05e-06, truncated_at=[4e-06, 6e-06]
                    )
                ),
            }
        )
        simulator = pybop.pybamm.Simulator(
            model, parameter_values=parameter_values, protocol=dataset
        )
        cost = pybop.SumSquaredError(dataset)
        fitting_problem = pybop.Problem(simulator, cost)
        with pytest.warns(UserWarning, match="more than 2 parameters"):
            pybop.plot.contour(fitting_problem)

    def test_nyquist(self, backend, figure_input):
        pybop.plot.use_backend(backend)
        # Define model
        model = pybamm.lithium_ion.SPM(options={"surface form": "differential"})
        parameter_values = model.default_parameter_values

        # Fitting parameters
        parameter_values.update(
            {
                "Positive electrode thickness [m]": pybop.Parameter(
                    distribution=pybop.Gaussian(
                        60e-6, 1e-6, truncated_at=[10e-6, 80e-6]
                    )
                ),
            }
        )

        # Form dataset
        dataset = pybop.Dataset(
            {
                "Frequency [Hz]": np.logspace(-4, 5, 10),
                "Current [A]": np.ones(10) * 0.0,
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
        fig, ax = figure_input
        inputs = problem.parameters.to_dict([60e-6])
        pybop.plot.nyquist(
            problem, inputs=inputs, title="Optimised Comparison", figures=fig, axes=[ax]
        )

        # Without inputs
        fig = pybop.plot.nyquist(problem, title="Optimised Comparison", show=False)
        assert fig is not None

    def test_util(self, backend, figure_input):
        # Test the utility functions
        pybop.plot.use_backend(backend)
        assert pybop.plot.remove_brackets(["Trace [1]", "Trace [2]"])[0] == "Trace  / 1"
        assert pybop.plot.remove_brackets(10) == 10

        x, y = pybop.plot.parse_data(np.zeros((3, 20)), np.zeros((3, 20)))
        assert (
            isinstance(x, list) and isinstance(y, list) and len(x) == 3 and len(y) == 3
        )

        if backend.lower() == "matplotlib":
            backend_inputs = [
                "matplotlib",
                "MaTpLoTliB",
                pybop.plot.backends.MatplotlibBackend(),
                None,
            ]
            figures_inputs = [None, [], [plt.figure()]]
            for backend_input in backend_inputs:
                for figures_input in figures_inputs:
                    backend_return = pybop.plot.get_backend_from_figure(
                        backend_input, figures_input
                    )
                    assert isinstance(
                        backend_return, pybop.plot.backends.MatplotlibBackend
                    )

        if backend.lower() == "plotly":
            backend_inputs = [
                "plotly",
                "PlOtLy",
                pybop.plot.backends.PlotlyBackend(),
                None,
            ]
            go = pybop.plot.backends.PlotlyManager().go
            figures_inputs = [None, [], [go.Figure()]]
            for backend_input in backend_inputs:
                for figures_input in figures_inputs:
                    backend_return = pybop.plot.get_backend_from_figure(
                        backend_input, figures_input
                    )
                    assert isinstance(backend_return, pybop.plot.backends.PlotlyBackend)

        # Assert error is raised for unsupported backend
        with pytest.raises(
            ModuleNotFoundError, match="Plotting backend nonsense is not available."
        ):
            backend = pybop.plot.get_backend_from_figure("nonsense", None)

        # Error if figure is not a plotly or matplotlib figure
        with pytest.raises(
            ValueError,
            match="Could not determine the backend from the provided figure of type <class 'object'>",
        ):
            backend = pybop.plot.get_backend_from_figure(None, [object()])

        # Assert current default used if backend is nonsensical
        with pytest.raises(
            ModuleNotFoundError,
            match="Plotting backend nonsense is not available. The current backend has not been updated. \n"
            f"The current backend is set to {backend}",
        ):
            pybop.plot.use_backend("nonsense")

        # Use figure's backend if backend is different from the figure's backend
        with pytest.warns(
            UserWarning,
            match="Backend wrong backend does not match the provided figure's backend",
        ):
            fig = figure_input[0]
            pybop.plot.get_backend_from_figure("wrong backend", [fig])

    def test_backend(self, backend):
        backend = pybop.plot.get_backend(backend)
        traces = []
        for i in range(3):
            traces.append(
                backend.line(
                    [1, 2],
                    [1 + i, 2 + i],
                    style=dict(
                        linestyle="solid",
                        marker="o",
                        xaxis_title=f"X-axis {i}",
                        yaxis_title=f"Y-axis {i}",
                    ),
                )
            )
        fig = backend.create_figure(
            title="Test Figure",
            xaxis_title="X-axis",
            yaxis_title="Y-axis",
            traces=traces,
        )

        # Test error handling for invalid inputs for parse_input_axes
        with pytest.raises(
            ValueError, match="This plot requires 5 axes. 2 axes provided."
        ):
            axes = (
                [fig.gca(), fig.gca()]
                if backend.name == "matplotlib"
                else [(1, 1), (1, 2)]
            )
            backend.parse_input_axes(fig, axes, num_plots=5, allow_single_axis=False)

        with pytest.raises(
            ValueError,
            match="This plot requires either 5 axes or a single axis. 2 axes provided.",
        ):
            axes = (
                [fig.gca(), fig.gca()]
                if backend.name == "matplotlib"
                else [(1, 1), (1, 2)]
            )
            backend.parse_input_axes(fig, axes, num_plots=5, allow_single_axis=True)

        with pytest.raises(
            ValueError,
            match="Please provide the same number of figures and axes or only one figure.",
        ):
            backend.parse_input_axes(
                [fig, fig],
                [fig.gca() if backend.name == "matplotlib" else (1, 1)],
                num_plots=2,
                allow_single_axis=False,
            )

        with pytest.warns(
            UserWarning, match="Axes argument ignored if no figure provided."
        ):
            backend.parse_input_axes(
                None,
                [fig.gca() if backend.name == "matplotlib" else (1, 1)],
                num_plots=1,
                allow_single_axis=False,
            )

        if backend.name == "plotly":
            with pytest.raises(ValueError, match="Axis must be a tuple"):
                backend.parse_input_axes(fig, (1, 3, 4))

        # Axes from figures
        figures, axes, create_figure, single_axis = backend.parse_input_axes(
            [fig, fig], None, num_plots=2, allow_single_axis=False
        )
        assert (
            len(figures) == 2
            and len(axes) == 2
            and create_figure is False
            and single_axis is False
        )

        # loc property for legend
        with pytest.raises(ValueError, match="loc property must consist of 2 keywords"):
            backend.legend(fig, style=dict(loc="upper"))

        # subplots with not enough grid space
        with pytest.raises(ValueError, match="Insufficient subplots"):
            backend.make_subplots(num_rows=1, num_cols=2, num_plots=5)

        # try plotting line without y data
        with pytest.raises(ValueError, match="y must be provided"):
            backend.line()

        # Some legend options
        backend.legend(fig, style=dict(outside=("right", 0.1)))
        backend.legend(fig, style=dict(outside=("left", 0.1)))
        backend.legend(fig, style=dict(outside=("top", 0.1)))
        backend.legend(fig, style=dict(outside=("bottom", 0.1)))

        # Some line styling
        backend.line(
            x=[1, 2],
            y=[1, 2],
            style=dict(linestyle="solid", marker="o", markeredgewidth=2.0),
        )
