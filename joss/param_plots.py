# A script to generate parameterisation plots for the JOSS paper.

import numpy as np
import plotly
import pybamm

import pybop
from pybop.plot import PlotlyManager

go = PlotlyManager().go
np.random.seed(8)

# Choose which plots to show and save
create_plot = {}
create_plot["simulation"] = True
create_plot["landscape"] = True
create_plot["minimising"] = True
create_plot["maximising"] = True
create_plot["gradient"] = True  # takes longest
create_plot["evolution"] = True
create_plot["heuristic"] = True
create_plot["posteriors"] = True


# Parameter set and model definition
parameter_set = pybop.ParameterSet.pybamm("Chen2020")
options = {"contact resistance": "true"}
parameter_set["Contact resistance [Ohm]"] = 0.01
solver = pybamm.IDAKLUSolver(rtol=1e-7, atol=1e-7)
model = pybop.lithium_ion.SPM(
    parameter_set=parameter_set, options=options, solver=solver
)

# Fitting parameters
parameters = pybop.Parameters(
    pybop.Parameter(
        "Negative particle diffusivity [m2.s-1]",
        initial_value=9e-14,
        prior=pybop.Gaussian(9e-14, 0.5e-14),
        transformation=pybop.LogTransformation(),
        bounds=[1.9e-14, 12e-14],
        true_value=parameter_set["Negative particle diffusivity [m2.s-1]"],
    ),
    pybop.Parameter(
        "Contact resistance [Ohm]",
        initial_value=0.02,
        prior=pybop.Gaussian(0.02, 0.005),
        # transformation=pybop.ScaledTransformation(coefficient=100),
        transformation=pybop.LogTransformation(),
        bounds=[0.005, 0.025],
        true_value=parameter_set["Contact resistance [Ohm]"],
    ),
)

# Generate input dataset
experiment = pybop.Experiment(
    [
        (
            "Discharge at 1C until 2.8 V (20 second period)",
            "Rest for 30 minutes (20 second period)",
        )
    ]
)

# Generate synthetic data and add Gaussian noise
solution = model.predict(experiment=experiment)
sigma = 0.005
values = solution["Voltage [V]"].data
corrupt_values = values + np.random.normal(0, sigma, len(values))

if create_plot["simulation"]:
    # Plot the data and the simulation
    simulation_plot_dict = pybop.plot.StandardPlot(
        x=solution["Time [s]"].data,
        y=[values, corrupt_values, solution["Battery open-circuit voltage [V]"].data],
        trace_names=[
            "Voltage [V]",
            "Voltage with Gaussian noise [V]",
            "Open-circuit voltage [V]",
        ],
    )
    simulation_plot_dict.traces[1].mode = "markers"
    simulation_fig = simulation_plot_dict(show=False)
    simulation_fig.update_layout(
        xaxis_title="Time / s",
        yaxis_title="Voltage / V",
        width=576,
        height=576,
    )
    simulation_fig.show()
    simulation_fig.write_image("joss/figures/simulation.png")

# Form dataset
dataset = pybop.Dataset(
    {
        "Time [s]": solution["Time [s]"].data,
        "Current function [A]": solution["Current [A]"].data,
        "Voltage [V]": corrupt_values,
    }
)

# Generate the fitting problem
problem = pybop.FittingProblem(model, parameters, dataset)

if create_plot["landscape"]:
    # Plot the cost landscape with the initial and true values
    cost = pybop.RootMeanSquaredError(problem)
    landscape_fig = pybop.plot.contour(
        cost,
        steps=25,
        show=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        title=None,
    )
    initial_value = parameters.initial_value()
    true_value = parameters.true_value()
    landscape_fig.add_trace(
        go.Scatter(
            x=[initial_value[0]],
            y=[initial_value[1]],
            mode="markers",
            marker_symbol="circle",
            marker=dict(
                color="mediumspringgreen",
                line_color="mediumspringgreen",
                line_width=1,
                size=14,
                showscale=False,
            ),
            name="Initial values",
        )
    )
    landscape_fig.add_trace(
        go.Scatter(
            x=[true_value[0]],
            y=[true_value[1]],
            mode="markers",
            marker_symbol="cross",
            marker=dict(
                color="white",
                line_color=None,
                line_width=1,
                size=14,
                showscale=False,
            ),
            name="True values",
        )
    )
    landscape_fig.show()
    landscape_fig.write_image("joss/figures/landscape.png")


# Categorise the costs
minimising_cost_classes = [
    pybop.Minkowski,  # largest
    pybop.SumSquaredError,
    pybop.SumofPower,
    pybop.RootMeanSquaredError,  # smallest
]
maximising_cost_classes = [
    pybop.GaussianLogLikelihood,
    pybop.GaussianLogLikelihoodKnownSigma,
    pybop.LogPosterior,
    pybop.LogPosterior,
]


if create_plot["minimising"]:
    # Show cost convergence using same optimiser for different cost functions
    convergence_traces = []
    for cost in minimising_cost_classes:
        # Define keyword arguments for the cost class
        kwargs = {}
        if cost is pybop.SumofPower:
            kwargs["p"] = 2.5

        # Define the cost and optimiser
        cost = cost(problem, **kwargs)
        optim = pybop.SciPyMinimize(
            cost,
            verbose=True,
            max_iterations=500,
            max_unchanged_iterations=25,
        )

        # Run optimisation
        results = optim.run()
        print("True parameter values:", parameters.true_value())

        # Plot convergence
        cost_log = optim.log["cost_best"]
        iteration_numbers = list(range(1, len(cost_log) + 1))
        convergence_plot_dict = pybop.plot.StandardPlot(
            x=iteration_numbers,
            y=cost_log,
            trace_names=type(cost).__name__,
            trace_options={"line": {"width": 4, "dash": "dash"}},
        )
        convergence_traces.extend(convergence_plot_dict.traces)

    # Plot minimising convergence traces together
    convergence_fig = go.Figure(
        data=convergence_traces,
        layout=dict(
            xaxis_title="Iteration",
            yaxis_title="Cost",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            plot_bgcolor="white",
            width=576,
            height=576,
        ),
    )
    convergence_fig.show()
    convergence_fig.write_image("joss/figures/convergence_minimising.png")


if create_plot["maximising"]:
    ## Do the same for the maximising cost functions
    convergence_traces = []
    first_MAP = True

    for cost in maximising_cost_classes:
        if cost is pybop.GaussianLogLikelihoodKnownSigma:
            cost = cost(problem, sigma0=sigma)
        elif cost is pybop.GaussianLogLikelihood:
            cost = cost(problem, sigma0=4 * sigma)
        elif cost is pybop.LogPosterior and first_MAP:
            cost = cost(
                log_likelihood=pybop.GaussianLogLikelihoodKnownSigma(
                    problem, sigma0=sigma
                )
            )
            first_MAP = False
        elif cost is pybop.LogPosterior:
            cost = cost(log_likelihood=pybop.GaussianLogLikelihood(problem))

        # Define the optimiser
        optim = pybop.CMAES(
            cost,
            verbose=True,
            max_iterations=125,
            max_unchanged_iterations=25,
        )

        # Run optimisation
        results = optim.run()
        print("True parameter values:", parameters.true_value())

        # Plot convergence
        cost_log = optim.log["cost_best"]
        iteration_numbers = list(range(1, len(cost_log) + 1))
        convergence_plot_dict = pybop.plot.StandardPlot(
            x=iteration_numbers,
            y=cost_log,
            trace_names=type(cost).__name__
            + " "
            + (
                type(cost.likelihood).__name__
                if isinstance(cost, pybop.LogPosterior)
                else ""
            ),
            trace_options={"line": {"width": 4, "dash": "dash"}},
        )
        convergence_traces.extend(convergence_plot_dict.traces)

    # Plot maximising convergence traces together
    convergence_fig = go.Figure(
        data=convergence_traces,
        layout=dict(
            xaxis_title="Iteration",
            yaxis_title="Likelihood",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            plot_bgcolor="white",
            width=576,
            height=576,
        ),
    )
    convergence_fig.show()
    convergence_fig.write_image("joss/figures/convergence_maximising.png")


# Categorise the optimisers
gradient_optimiser_classes = [
    pybop.AdamW,
    pybop.GradientDescent,
    pybop.IRPropMin,
    pybop.SciPyMinimize,  # with L-BFGS-B and jac=True
]
evolution_optimiser_classes = [
    pybop.SciPyDifferentialEvolution,  # most iterations
    pybop.XNES,
    pybop.CMAES,
    pybop.SNES,  # least iterations
]
heuristic_optimiser_classes = [
    pybop.PSO,
    pybop.NelderMead,
    pybop.CuckooSearch,
    # pybop.SciPyMinimize,  # with NelderMead
]


if create_plot["gradient"]:
    ## Show parameter convergence for difference optimisers and same cost function
    parameter_traces = []
    for i, optimiser in enumerate(gradient_optimiser_classes):
        # Define keyword arguments for the optimiser class
        kwargs = {"sigma0": [0.08, 0.05]}
        if optimiser is pybop.SciPyMinimize:
            kwargs["method"] = "L-BFGS-B"
            kwargs["jac"] = True
            kwargs["tol"] = 1e-9
        elif optimiser is pybop.GradientDescent:
            kwargs["sigma0"] = [1.2, 0.3]
        elif optimiser is pybop.AdamW:
            kwargs["sigma0"] = [0.25, 0.08]
        # elif optimiser is pybop.IRPropMin:
        #     kwargs["sigma0"] = [4e-3,2e-2]

        # Define the cost and optimiser
        cost = pybop.SumSquaredError(problem)
        optim = optimiser(
            cost,
            verbose=True,
            max_evaluations=60,
            maxfev=60,
            max_unchanged_iterations=60,
            **kwargs,
        )

        # Run optimisation
        results = optim.run()
        print("True parameter values:", parameters.true_value())

        # Plot the parameter traces
        parameter_fig = pybop.plot.parameters(optim, show=False, title=None)
        parameter_fig.update_traces(dict(mode="markers"))
        colours = parameter_fig.layout.template.layout.colorway
        for j in range(len(parameter_fig.data)):
            parameter_fig.data[j].name = optim.name()
            parameter_fig.data[j].marker.color = colours[i]
            if j > 0:
                parameter_fig.data[j].showlegend = False
        parameter_traces.extend(parameter_fig.data)

        # Plot the cost landscape with optimisation path
        contour = pybop.plot.contour(
            optim, steps=25, title="", margin=dict(l=30, r=30, t=30, b=30)
        )
        contour.write_image(f"joss/figures/contour_gradient_{i}.png")

    # Plot the parameter traces together
    parameter_fig.update_layout(width=576, height=1024, plot_bgcolor="white")
    parameter_fig.data = []
    parameter_fig.add_traces(parameter_traces)
    parameter_fig = plotly.subplots.make_subplots(figure=parameter_fig, rows=2, cols=1)
    parameter_fig.show()
    parameter_fig.write_image("joss/figures/gradient_parameters.png")


if create_plot["evolution"]:
    ## Do the same for the evolution strategies
    parameter_traces = []
    for i, optimiser in enumerate(evolution_optimiser_classes):
        # Define keyword arguments for the optimiser class
        kwargs = {}

        # Define the cost and optimiser
        cost = pybop.SumSquaredError(problem)
        if optimiser is pybop.SciPyDifferentialEvolution:
            optim = optimiser(
                cost,
                verbose=True,
                max_iterations=50,
                max_unchanged_iterations=25,
                popsize=3,
                **kwargs,
            )
        else:
            optim = optimiser(
                cost,
                verbose=True,
                max_iterations=300,
                max_unchanged_iterations=300,
                max_evaluations=338,
                popsize=6,
                **kwargs,
            )

        # Run optimisation
        results = optim.run()
        print("True parameter values:", parameters.true_value())

        # Plot the parameter traces
        parameter_fig = pybop.plot.parameters(optim, show=False, title=None)
        parameter_fig.update_traces(dict(mode="markers"))
        colours = parameter_fig.layout.template.layout.colorway
        for j in range(len(parameter_fig.data)):
            parameter_fig.data[j].name = optim.name()
            parameter_fig.data[j].marker.color = colours[i]
            if j > 0:
                parameter_fig.data[j].showlegend = False
        parameter_traces.extend(parameter_fig.data)

        # Plot the cost landscape with optimisation path
        contour = pybop.plot.contour(
            optim, steps=25, title="", margin=dict(l=30, r=30, t=30, b=30)
        )
        contour.write_image(f"joss/figures/contour_evolution_{i}.png")

    # Plot the parameter traces together
    parameter_fig.update_layout(width=576, height=1024, plot_bgcolor="white")
    parameter_fig.data = []
    parameter_fig.add_traces(parameter_traces)
    parameter_fig = plotly.subplots.make_subplots(figure=parameter_fig, rows=2, cols=1)
    parameter_fig.show()
    parameter_fig.write_image("joss/figures/evolution_parameters.png")


if create_plot["heuristic"]:
    ## Do the same for the (meta)heuristics
    parameter_traces = []
    for i, optimiser in enumerate(heuristic_optimiser_classes):
        # Define keyword arguments for the optimiser class
        kwargs = {}

        # Define the cost and optimiser
        cost = pybop.SumSquaredError(problem)
        optim = optimiser(
            cost,
            verbose=True,
            sigma0=0.02,
            max_iterations=500,
            max_unchanged_iterations=25,
            max_evaluations=300,
            **kwargs,
        )

        # Run optimisation
        results = optim.run()
        print("True parameter values:", parameters.true_value())

        # Plot the parameter traces
        parameter_fig = pybop.plot.parameters(optim, show=False, title=None)
        parameter_fig.update_traces(dict(mode="markers"))
        colours = parameter_fig.layout.template.layout.colorway
        for j in range(len(parameter_fig.data)):
            parameter_fig.data[j].name = optim.name()
            parameter_fig.data[j].marker.color = colours[i]
            if j > 0:
                parameter_fig.data[j].showlegend = False
        parameter_traces.extend(parameter_fig.data)

        # Plot the cost landscape with optimisation path
        contour = pybop.plot.contour(
            optim, steps=25, title="", margin=dict(l=30, r=30, t=30, b=30)
        )
        contour.write_image(f"joss/figures/contour_heuristic_{i}.png")

    # Plot the parameter traces together
    parameter_fig.update_layout(width=576, height=1024, plot_bgcolor="white")
    parameter_fig.data = []
    parameter_fig.add_traces(parameter_traces)
    parameter_fig = plotly.subplots.make_subplots(figure=parameter_fig, rows=2, cols=1)
    parameter_fig.show()
    parameter_fig.write_image("joss/figures/heuristic_parameters.png")


if create_plot["posteriors"]:
    likelihood = pybop.GaussianLogLikelihood(problem)
    posterior = pybop.LogPosterior(likelihood)

    sampler = pybop.HaarioBardenetACMC(
        posterior,
        chains=5,
        verbose=True,
        max_iterations=3500,
        warm_up=1500,
        parallel=True,
    )

    samples = sampler.run()
    summary = pybop.PosteriorSummary(samples)
    print(summary.rhat())
    print(summary.effective_sample_size(mixed_chains=True))
    summary.plot_trace()
    summary.plot_chains()
    summary.plot_posterior()
