# A script to generate parameterisation plots for the JOSS paper.

import time

import matplotlib.pyplot as plt
import numpy as np
import plotly
import pybamm
from matplotlib.ticker import ScalarFormatter

import pybop
from pybop.plot import PlotlyManager

go = PlotlyManager().go
px = PlotlyManager().px
make_subplots = PlotlyManager().make_subplots
np.random.seed(8)

# Choose which plots to show and save
create_plot = {}
create_plot["simulation"] = True
create_plot["landscape"] = True
create_plot["minimising"] = True
create_plot["maximising"] = True
create_plot["gradient"] = True
create_plot["evolution"] = True
create_plot["heuristic"] = True
create_plot["posteriors"] = True
create_plot["eis"] = True

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
        y=[corrupt_values, solution["Battery open-circuit voltage [V]"].data, values],
        trace_names=[
            "Voltage w/ Gaussian noise [V]",
            "Open-circuit voltage [V]",
            "Voltage [V]",
        ],
    )
    simulation_plot_dict.traces[0].mode = "markers"
    simulation_fig = simulation_plot_dict(show=False)
    simulation_fig.update_layout(
        xaxis_title="Time / s",
        yaxis_title="Voltage / V",
        width=576,
        height=576,
        legend_traceorder="reversed",
    )
    simulation_fig.show()
    simulation_fig.write_image("joss/figures/simulation.pdf")
    time.sleep(3)
    simulation_fig.write_image("joss/figures/simulation.pdf")

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
            marker_symbol="x",
            marker=dict(
                color="white",
                line_color="black",
                line_width=1,
                size=16,
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
                color="black",
                line_color="white",
                line_width=1,
                size=16,
                showscale=False,
            ),
            name="True values",
        )
    )
    landscape_fig.show()
    landscape_fig.write_image("joss/figures/landscape.pdf")


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
            trace_names=cost.name,
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
            yaxis=dict(
                gridcolor=px.colors.qualitative.Pastel2[7],
                zerolinecolor=px.colors.qualitative.Pastel2[7],
                zerolinewidth=1,
                titlefont_size=14,
                tickfont_size=14,
            ),
            xaxis=dict(titlefont_size=14, tickfont_size=14),
            width=600,
            height=600,
        ),
    )
    convergence_fig.show()
    convergence_fig.write_image("joss/figures/convergence_minimising.pdf")


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
            trace_names=cost.name
            + " "
            + (cost.likelihood.name if isinstance(cost, pybop.LogPosterior) else ""),
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
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font_size=14,
            ),
            plot_bgcolor="white",
            yaxis=dict(
                gridcolor=px.colors.qualitative.Pastel2[7],
                zerolinecolor=px.colors.qualitative.Pastel2[7],
                zerolinewidth=1,
                titlefont_size=14,
                tickfont_size=14,
            ),
            xaxis=dict(titlefont_size=14, tickfont_size=14),
            width=600,
            height=600,
        ),
    )
    convergence_fig.show()
    convergence_fig.write_image("joss/figures/convergence_maximising.pdf")


# Categorise the optimisers
gradient_optimiser_classes = [
    pybop.AdamW,
    pybop.IRPropMin,
    pybop.GradientDescent,
    pybop.SciPyMinimize,  # with L-BFGS-B and jac=True
]
evolution_optimiser_classes = [
    pybop.CMAES,
    pybop.XNES,
    pybop.SNES,
    pybop.SciPyDifferentialEvolution,
]
heuristic_optimiser_classes = [
    pybop.PSO,
    pybop.NelderMead,
    pybop.CuckooSearch,
    # pybop.SciPyMinimize,  # with NelderMead
]

# Define the cost
cost = pybop.RootMeanSquaredError(problem)

# Create subplot figure
max_optims = max(
    len(gradient_optimiser_classes),
    len(evolution_optimiser_classes),
    len(heuristic_optimiser_classes),
)
subplot_contour_fig = make_subplots(
    rows=3,
    cols=max_optims,
    shared_yaxes=True,
    shared_xaxes=True,
    x_title="Negative particle diffusivity [m2.s-1]",
    y_title="Contact resistance [Ohm]",
    horizontal_spacing=0.025,
    vertical_spacing=0.04,
    subplot_titles=[
        cls.__name__
        for cls in [
            *gradient_optimiser_classes,
            *evolution_optimiser_classes,
            *heuristic_optimiser_classes,
        ]
    ],
)

if create_plot["gradient"]:
    # Create subplot structure
    num_optimisers = len(gradient_optimiser_classes)
    parameter_traces = []

    for i, optimiser in enumerate(gradient_optimiser_classes):
        # Define keyword arguments for the optimiser class
        kwargs = {"sigma0": [0.08, 0.05]}
        if optimiser is pybop.SciPyMinimize:
            kwargs["method"] = "L-BFGS-B"
            kwargs["jac"] = True
            kwargs["tol"] = 1e-9
        elif optimiser is pybop.GradientDescent:
            kwargs["sigma0"] = [11, 2.25]
        elif optimiser is pybop.AdamW:
            kwargs["sigma0"] = 0.2

        # Construct the optimiser
        optim = optimiser(
            cost,
            verbose=True,
            max_evaluations=50,
            maxfev=50,
            max_unchanged_iterations=50,
            **kwargs,
        )

        if optimiser is pybop.AdamW:
            optim.optimiser.b1 = 0.8
            optim.optimiser.b2 = 0.9
            optim.optimiser.lam = 0.01

        # Run optimisation
        optim = optimiser(cost, **kwargs)
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

        # Collect contour plot data
        contour = pybop.plot.contour(
            optim,
            steps=25,
            title="",
            showlegend=False,
            show=False,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        if i == num_optimisers - 1:
            contour.update_traces(
                showscale=(i == num_optimisers - 1),
                colorbar=dict(
                    title=dict(
                        text="Cost",
                        # side="right",
                        font=dict(size=18),
                    ),
                    tickfont=dict(size=16),
                ),
                selector=dict(type="contour"),
            )
        else:
            contour.update_traces(
                showscale=False,
                selector=dict(type="contour"),
            )

        # Add all traces from the contour figure to the subplot
        for trace in contour.data:
            subplot_contour_fig.add_trace(trace, row=1, col=i + 1)

    # Plot the parameter traces together
    parameter_fig.update_layout(
        width=576,
        height=1024,
        plot_bgcolor="white",
        yaxis=dict(
            gridcolor=px.colors.qualitative.Pastel2[7],
            zerolinecolor=px.colors.qualitative.Pastel2[7],
            zerolinewidth=1,
            titlefont_size=14,
            tickfont_size=14,
        ),
        xaxis=dict(titlefont_size=14, tickfont_size=14),
        yaxis2=dict(
            gridcolor=px.colors.qualitative.Pastel2[7],
            zerolinecolor=px.colors.qualitative.Pastel2[7],
            zerolinewidth=1,
            titlefont_size=14,
            tickfont_size=14,
        ),
        xaxis2=dict(titlefont_size=14, tickfont_size=14),
    )
    parameter_fig.data = []
    parameter_fig.add_traces(parameter_traces)
    parameter_fig = plotly.subplots.make_subplots(figure=parameter_fig, rows=2, cols=1)
    parameter_fig.show()
    parameter_fig.write_image("joss/figures/gradient_parameters.pdf")

if create_plot["evolution"]:
    # Create subplot structure
    num_optimisers = len(evolution_optimiser_classes)

    # Define shared optimiser settings
    default_kwargs = {
        "verbose": True,
        "max_iterations": 300,
        "max_unchanged_iterations": 300,
        "max_evaluations": 338,
        "popsize": 6,
    }
    parameter_traces = []

    for i, optimiser in enumerate(evolution_optimiser_classes):
        # Set specific arguments for SciPyDifferentialEvolution
        kwargs = default_kwargs.copy()
        if optimiser is pybop.SciPyDifferentialEvolution:
            kwargs.update(
                {"max_iterations": 50, "max_unchanged_iterations": 25, "popsize": 3}
            )

        # Run optimisation
        optim = optimiser(cost, **kwargs)
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

        # Collect contour plot data
        contour = pybop.plot.contour(
            optim,
            steps=25,
            title="",
            showlegend=False,
            show=False,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        contour.update_traces(
            showscale=False,
            selector=dict(type="contour"),
        )
        # Add all traces from the contour figure to the subplot
        for trace in contour.data:
            subplot_contour_fig.add_trace(trace, row=2, col=i + 1)

    # Plot the parameter traces together
    parameter_fig.update_layout(
        width=576,
        height=1024,
        plot_bgcolor="white",
        yaxis=dict(
            gridcolor=px.colors.qualitative.Pastel2[7],
            zerolinecolor=px.colors.qualitative.Pastel2[7],
            zerolinewidth=1,
            titlefont_size=14,
            tickfont_size=14,
        ),
        xaxis=dict(titlefont_size=14, tickfont_size=14),
        yaxis2=dict(
            gridcolor=px.colors.qualitative.Pastel2[7],
            zerolinecolor=px.colors.qualitative.Pastel2[7],
            zerolinewidth=1,
            titlefont_size=14,
            tickfont_size=14,
        ),
        xaxis2=dict(titlefont_size=14, tickfont_size=14),
    )
    parameter_fig.data = []
    parameter_fig.add_traces(parameter_traces)
    parameter_fig = plotly.subplots.make_subplots(figure=parameter_fig, rows=2, cols=1)
    parameter_fig.show()
    parameter_fig.write_image("joss/figures/evolution_parameters.pdf")


if create_plot["heuristic"]:
    # Create subplot structure
    num_optimisers = len(evolution_optimiser_classes)

    # Define shared optimiser settings
    default_kwargs = {
        "verbose": True,
        "max_iterations": 300,
        "max_unchanged_iterations": 300,
        "max_evaluations": 338,
        "popsize": 6,
    }
    parameter_traces = []
    for i, optimiser in enumerate(heuristic_optimiser_classes):
        # Define keyword arguments for the optimiser class
        kwargs = {}

        # Construct the optimiser
        optim = optimiser(
            cost,
            verbose=True,
            sigma0=0.05,
            max_iterations=150,
            max_unchanged_iterations=150,
            max_evaluations=150,
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

        # Collect contour plot data
        contour = pybop.plot.contour(
            optim,
            steps=25,
            title="",
            showlegend=False,
            show=False,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        contour.update_traces(
            showscale=False,
            selector=dict(type="contour"),
        )

        # Add all traces from the contour figure to the subplot
        for trace in contour.data:
            subplot_contour_fig.add_trace(trace, row=3, col=i + 1)

    # Update layout to configure the color bar and plot dimensions
    bounds = cost.parameters.get_bounds_for_plotly()
    subplot_contour_fig.update_xaxes(
        dict(titlefont_size=12, tickfont_size=16, range=bounds[0])
    )
    subplot_contour_fig.update_yaxes(
        dict(titlefont_size=12, tickfont_size=16, range=bounds[1])
    )
    subplot_contour_fig.update_layout(
        showlegend=False,
        height=400 * 3,
        width=400 * max_optims,
    )
    subplot_contour_fig.update_annotations(font_size=18)
    subplot_contour_fig.update_annotations(
        x=-0.01, font_size=18, selector={"text": "Contact resistance [Ohm]"}
    )

    # Show figure and save image
    subplot_contour_fig.show()
    subplot_contour_fig.write_image("joss/figures/joss/contour_subplot.pdf")

    # Plot the parameter traces together
    parameter_fig.update_layout(
        width=576,
        height=1024,
        plot_bgcolor="white",
        yaxis=dict(
            gridcolor=px.colors.qualitative.Pastel2[7],
            zerolinecolor=px.colors.qualitative.Pastel2[7],
            zerolinewidth=1,
            titlefont_size=14,
            tickfont_size=14,
        ),
        xaxis=dict(titlefont_size=14, tickfont_size=14),
        yaxis2=dict(
            gridcolor=px.colors.qualitative.Pastel2[7],
            zerolinecolor=px.colors.qualitative.Pastel2[7],
            zerolinewidth=1,
            titlefont_size=14,
            tickfont_size=14,
        ),
        xaxis2=dict(titlefont_size=14, tickfont_size=14),
    )
    parameter_fig.data = []
    parameter_fig.add_traces(parameter_traces)
    parameter_fig = plotly.subplots.make_subplots(figure=parameter_fig, rows=2, cols=1)
    parameter_fig.update_layout(yaxis1=dict(range=[1e-14, 10e-14]))
    parameter_fig.show()
    parameter_fig.write_image("joss/figures/heuristic_parameters.pdf")


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

    # Enable LaTeX for text rendering
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",  # Use a LaTeX-compatible font
            "font.serif": ["Computer Modern"],  # Default LaTeX font
        }
    )

    # Sample from the distributions
    neg_particle_samples = summary.all_samples[:, 0]
    contact_samples = summary.all_samples[:, 1]
    sigma_samples = summary.all_samples[:, 2]

    # Create a grid for subplots
    fig = plt.figure(figsize=(13, 6))
    gs = fig.add_gridspec(1, 3, height_ratios=[1])

    # Function to format axis
    def format_axis(ax):
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-1, 1))  # Set limits to use scientific notation
        formatter.set_scientific(True)  # Ensure scientific notation
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
        ax.tick_params(axis="both", which="major", labelsize=14)

    # Top subplot for Neg Particle
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(
        neg_particle_samples,
        bins=50,
        density=False,
        alpha=0.6,
        color="tab:red",
        label="$R_n$",
    )
    ax1.set_title("Negative Particle Radius", fontsize=22)
    ax1.set_xlabel(r"m", fontsize=18)
    ax1.set_ylabel("Density", fontsize=18)
    ax1.set_xlim(
        parameter_set["Negative particle diffusivity [m2.s-1]"] * 0.95,
        parameter_set["Negative particle diffusivity [m2.s-1]"] * 1.07,
    )
    # ax1.set_ylim(0, 1e4)
    ax1.tick_params(axis="both", which="major", labelsize=18)
    ax1.axvspan(
        summary.get_summary_statistics()[("ci_lower")][0],
        summary.get_summary_statistics()[("ci_upper")][0],
        alpha=0.1,
        color="tab:red",
    )
    format_axis(ax1)

    # Top right subplot for Contact Resistance
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(
        contact_samples,
        bins=50,
        density=False,
        alpha=0.6,
        color="tab:blue",
        label="$R_c$",
    )
    ax2.set_title("Contact Resistance", fontsize=22)
    ax2.set_xlabel(r"$\Omega$", fontsize=18)
    ax2.set_xlim(
        parameter_set["Contact resistance [Ohm]"] * 0.95,
        parameter_set["Contact resistance [Ohm]"] * 1.05,
    )
    # ax2.set_ylim(0, 1e3)
    ax2.tick_params(axis="both", which="major", labelsize=18)
    ax2.axvspan(
        summary.get_summary_statistics()[("ci_lower")][1],
        summary.get_summary_statistics()[("ci_upper")][1],
        alpha=0.1,
        color="tab:blue",
    )
    format_axis(ax2)

    # Bottom right subplot for sigma
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(
        sigma_samples,
        bins=50,
        density=False,
        alpha=0.6,
        color="tab:purple",
        label=r"$\sigma$",
    )
    ax3.set_title(r"Observation Noise, $\sigma$", fontsize=22)
    ax3.set_xlabel("V", fontsize=18)
    ax3.set_xlim(4e-3, 5.5e-3)
    # ax3.set_ylim(0, 10 * 1.1)
    ax3.tick_params(axis="both", which="major", labelsize=18)
    ax3.axvspan(
        summary.get_summary_statistics()[("ci_lower")][2],
        summary.get_summary_statistics()[("ci_upper")][2],
        alpha=0.1,
        color="tab:purple",
    )
    format_axis(ax3)

    # Adjust layout
    plt.tight_layout()
    plt.savefig("joss/figures/joss/posteriors.pdf")
    plt.show()


if create_plot["eis"]:

    def noise(sigma, values):
        # Generate real part noise
        real_noise = np.random.normal(0, sigma, values)

        # Generate imaginary part noise
        imag_noise = np.random.normal(0, sigma, values)

        # Combine them into a complex noise
        return real_noise + 1j * imag_noise

    var = pybamm.standard_spatial_vars
    var_pts = {var.x_n: 10, var.x_s: 10, var.x_p: 10, var.r_n: 30, var.r_p: 30}

    # Construct model
    model = pybop.lithium_ion.SPMe(
        parameter_set=parameter_set,
        eis=True,
        options={"surface form": "differential", "contact resistance": "true"},
        var_pts=var_pts,
    )

    initial_state = {"Initial SoC": 0.05}
    n_frequency = 35
    sigma0 = 5e-4
    f_eval = np.logspace(-3.75, 4, n_frequency)

    # Create synthetic data for parameter inference
    sim = model.simulateEIS(
        inputs={
            "Negative particle diffusivity [m2.s-1]": parameter_set[
                "Negative particle diffusivity [m2.s-1]"
            ],
            "Contact resistance [Ohm]": 0.01,
        },
        f_eval=f_eval,
        initial_state=initial_state,
    )

    # Form dataset
    dataset = pybop.Dataset(
        {
            "Frequency [Hz]": f_eval,
            "Current function [A]": np.zeros(len(f_eval)),
            "Impedance": sim["Impedance"] + noise(sigma0, len(sim["Impedance"])),
        }
    )

    signal = ["Impedance"]
    problem = pybop.FittingProblem(model, parameters, dataset, signal=signal)
    cost = pybop.RootMeanSquaredError(problem)

    optim = pybop.CMAES(cost, max_iterations=75, min_iterations=75, sigma0=0.25)
    results = optim.run()

    parameter_fig = pybop.plot.nyquist(problem, results.x, title="")
    parameter_fig[0].write_image("joss/figures/impedance_spectrum.pdf")

    landscape_fig = pybop.plot.contour(cost, steps=30, title="")
    initial_value = parameters.initial_value()
    true_value = parameters.true_value()
    landscape_fig.add_trace(
        go.Scatter(
            x=[initial_value[0]],
            y=[initial_value[1]],
            mode="markers",
            marker_symbol="x",
            marker=dict(
                color="white",
                line_color="black",
                line_width=1,
                size=16,
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
                color="black",
                line_color="white",
                line_width=1,
                size=16,
                showscale=False,
            ),
            name="True values",
        )
    )
    landscape_fig.show()
    landscape_fig.write_image("joss/figures/impedance_contour.pdf")
