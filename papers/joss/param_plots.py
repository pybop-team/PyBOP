# A script to generate parameterisation plots for the JOSS paper.


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
plt.rcParams.update({"text.usetex": True})  # Enable LaTeX
np.random.seed(8)  # Set random seed for reproducibility

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

# Define model, solver and parameter values
model = pybamm.lithium_ion.SPM(options={"contact resistance": "true"})
solver = pybamm.IDAKLUSolver(rtol=1e-7, atol=1e-7)
parameter_values = pybamm.ParameterValues("Chen2020")
parameter_values["Contact resistance [Ohm]"] = 0.01

# Define experiment
experiment = pybamm.Experiment(
    [
        "Discharge at 1C until 2.8 V (20 second period)",
        "Rest for 30 minutes (20 second period)",
    ]
)

# Generate a synthetic dataset with Gaussian noise
sigma = 0.005
solution = pybamm.Simulation(
    model, parameter_values=parameter_values, experiment=experiment, solver=solver
).solve()
values = solution["Voltage [V]"].data
corrupt_values = values + np.random.normal(0, sigma, len(values))

if create_plot["simulation"]:
    # Plot the data and the simulation
    simulation_plot_dict = pybop.plot.StandardPlot(
        x=solution["Time [s]"].data,
        y=[corrupt_values, solution["Battery open-circuit voltage [V]"].data, values],
        trace_names=[
            "Voltage with noise",
            "Open-circuit voltage",
            "Voltage",
        ],
    )
    simulation_plot_dict.traces[0].mode = "markers"
    simulation_fig = simulation_plot_dict(show=False)
    simulation_fig.update_layout(
        width=595,
        height=595,
        xaxis=dict(
            title=dict(text="Time / s", font_size=16),
            showline=True,
            linewidth=2,
            linecolor="black",
            mirror=True,
            tickfont_size=15,
        ),
        yaxis=dict(
            title=dict(text="Voltage / V", font_size=16),
            showline=True,
            linewidth=2,
            linecolor="black",
            mirror=True,
            tickfont_size=15,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=15),
        ),
        margin=dict(r=50, t=50),
    )
    simulation_fig.show()
    simulation_fig.write_image("figures/individual/simulation.pdf")

# Form dataset
dataset = pybop.Dataset(
    {
        "Time [s]": solution["Time [s]"].data,
        "Current function [A]": solution["Current [A]"].data,
        "Voltage [V]": corrupt_values,
    }
)

# Fitting parameters
true_value = [
    parameter_values[p]
    for p in ["Contact resistance [Ohm]", "Negative particle diffusivity [m2.s-1]"]
]
initial_value = [0.02, 9e-14]
parameter_values.update(
    {
        "Contact resistance [Ohm]": pybop.Parameter(
            initial_value=initial_value[0],
            prior=pybop.Gaussian(0.02, 0.005),
            transformation=pybop.ScaledTransformation(coefficient=200),
            bounds=[0.005, 0.025],
        ),
        "Negative particle diffusivity [m2.s-1]": pybop.Parameter(
            initial_value=initial_value[1],
            prior=pybop.Gaussian(9e-14, 2e-14),
            transformation=pybop.LogTransformation(),
            bounds=[1.9e-14, 12e-14],
        ),
    }
)

# Build the problem
simulator = pybop.pybamm.Simulator(
    model, parameter_values=parameter_values, protocol=dataset, solver=solver
)
cost = pybop.SumSquaredError(dataset)
problem = pybop.Problem(simulator, cost)

if create_plot["landscape"]:
    # Plot the cost landscape with the initial and true values
    landscape_fig = pybop.plot.contour(
        problem,
        steps=25,
        title=None,
        show=False,
        xaxis=dict(
            title=dict(text="Contact resistance / Ω", font_size=16), tickfont_size=15
        ),
        yaxis=dict(
            title=dict(
                text="Negative particle diffusivity / m<sup>2</sup>&#8239;s<sup>-1</sup>",
                font_size=16,
                standoff=0,
            ),
            tickfont_size=15,
            exponentformat="power",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=15),
        ),
        coloraxis_colorbar=dict(tickfont=dict(size=18)),
        margin=dict(t=50),
    )
    landscape_fig.update_traces(colorbar=dict(tickfont=dict(size=15)))
    landscape_fig.add_trace(
        go.Scatter(
            x=[initial_value[0]],
            y=[initial_value[1]],
            mode="markers",
            marker=dict(
                symbol="x",
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
            marker=dict(
                symbol="cross",
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
    landscape_fig.write_image("figures/individual/landscape.pdf")

# Categorise the costs
minimising_cost_classes = [
    pybop.Minkowski,  # largest
    pybop.SumSquaredError,
    pybop.SumOfPower,
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
        if cost is pybop.SumOfPower:
            kwargs["p"] = 2.5

        # Define the problem and optimiser
        cost = cost(dataset, **kwargs)
        problem = pybop.Problem(simulator, cost)
        options = pybop.SciPyMinimizeOptions(maxiter=50, method="BFGS", jac=True)
        optim = pybop.SciPyMinimize(problem, options=options)

        # Run optimisation
        result = optim.run()
        print(result)
        print("True parameter values:", true_value)

        # Plot convergence
        cost_log = result.cost
        iteration_numbers = list(range(1, len(cost_log) + 1))
        convergence_plot_dict = pybop.plot.StandardPlot(
            x=iteration_numbers,
            y=cost_log,
            trace_names=[cost.name],
            trace_options={"line": {"width": 4, "dash": "dash"}},
        )
        convergence_traces.extend(convergence_plot_dict.traces)

    # Plot minimising convergence traces together
    convergence_fig = go.Figure(
        data=convergence_traces,
        layout=dict(
            xaxis=dict(title=dict(text="Evaluation", font_size=16), tickfont_size=15),
            yaxis=dict(
                title=dict(text="Cost", font_size=16),
                gridcolor=px.colors.qualitative.Pastel2[7],
                zerolinecolor=px.colors.qualitative.Pastel2[7],
                zerolinewidth=1,
                tickfont_size=15,
            ),
            legend=dict(
                yanchor="top",
                y=0.95,
                xanchor="right",
                x=0.99,
                font_size=14,
                bordercolor="black",
                borderwidth=1,
            ),
            plot_bgcolor="white",
            width=600,
            height=600,
        ),
    )
    convergence_fig.update_traces(dict(mode="markers"))
    convergence_fig.show()
    convergence_fig.write_image("figures/individual/convergence_minimising.pdf")


if create_plot["maximising"]:
    ## Do the same for the maximising cost functions
    convergence_traces = []
    first_MAP = True

    for cost in maximising_cost_classes:
        if cost is pybop.GaussianLogLikelihoodKnownSigma:
            cost = cost(dataset, sigma0=sigma)
        elif cost is pybop.GaussianLogLikelihood:
            cost = cost(dataset, sigma0=4 * sigma)
        elif cost is pybop.LogPosterior and first_MAP:
            cost = cost(
                log_likelihood=pybop.GaussianLogLikelihoodKnownSigma(
                    dataset, sigma0=sigma
                )
            )
            first_MAP = False
        elif cost is pybop.LogPosterior:
            cost = cost(log_likelihood=pybop.GaussianLogLikelihood(dataset))

        # Define the problem and optimiser
        problem = pybop.Problem(simulator, cost)
        options = pybop.SciPyMinimizeOptions(maxiter=50, method="BFGS", jac=True)
        optim = pybop.SciPyMinimize(problem, options=options)

        # Run optimisation
        result = optim.run()
        print(result)
        print("True parameter values:", true_value)

        # Plot convergence
        cost_log = result.cost
        iteration_numbers = list(range(1, len(cost_log) + 1))
        convergence_plot_dict = pybop.plot.StandardPlot(
            x=iteration_numbers,
            y=cost_log,
            trace_names=cost.name
            + " "
            + (
                cost.log_likelihood.name if isinstance(cost, pybop.LogPosterior) else ""
            ),
            trace_options={"line": {"width": 4, "dash": "dash"}},
        )
        convergence_traces.extend(convergence_plot_dict.traces)

    # Plot maximising convergence traces together
    convergence_fig = go.Figure(
        data=convergence_traces,
        layout=dict(
            xaxis=dict(title=dict(text="Evaluation", font_size=16), tickfont_size=15),
            yaxis=dict(
                title=dict(text="Likelihood", font_size=18),
                gridcolor=px.colors.qualitative.Pastel2[7],
                zerolinecolor=px.colors.qualitative.Pastel2[7],
                zerolinewidth=1,
                tickfont_size=15,
            ),
            legend=dict(
                yanchor="bottom",
                y=0.15,
                xanchor="right",
                x=0.99,
                font_size=14,
                bordercolor="black",
                borderwidth=1,
            ),
            plot_bgcolor="white",
            width=600,
            height=600,
        ),
    )
    convergence_fig.update_traces(dict(mode="markers"))
    convergence_fig.show()
    convergence_fig.write_image("figures/individual/convergence_maximising.pdf")


# Categorise the optimisers
gradient_optimiser_classes = [
    pybop.AdamW,
    pybop.IRPropMin,
    pybop.GradientDescent,
    pybop.SciPyMinimize,  # with BFGS and jac=True
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
    pybop.SimulatedAnnealing,
]

# Define the problem
cost = pybop.RootMeanSquaredError(dataset)
problem = pybop.Problem(simulator, cost)

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
    x_title="Contact resistance / Ω",
    y_title="Negative particle diffusivity / m<sup>2</sup>&#8239;s<sup>-1</sup>",
    horizontal_spacing=0.03,
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
bounds = problem.parameters.get_bounds_for_plotly()

if create_plot["gradient"]:
    # Create subplot structure
    num_optimisers = len(gradient_optimiser_classes)
    parameter_traces = []

    for i, optimiser in enumerate(gradient_optimiser_classes):
        # Set optimiser options
        if optimiser is pybop.SciPyMinimize:
            options = pybop.SciPyMinimizeOptions(method="BFGS", jac=True, maxiter=50)
        else:
            options = pybop.PintsOptions(
                max_unchanged_iterations=50, max_evaluations=250
            )

        # Construct the optimiser
        optim = optimiser(problem, options=options)

        if optimiser is pybop.AdamW:
            optim.optimiser.b1 = 0.85
            optim.optimiser.b2 = 0.9
            optim.optimiser.lam = 0.005
        if optimiser is pybop.GradientDescent:
            optim.optimiser.set_learning_rate(eta=[11, 4.5])

        # Run optimisation
        print(optim.name)
        result = optim.run()
        print(result)
        print("True parameter values:", true_value)

        # Plot the parameter traces
        parameter_fig = pybop.plot.parameters(result, show=False, title=None)
        parameter_fig.update_layout(
            legend=dict(
                orientation="v",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=-0.05,
                font_size=1,
            )
        )
        parameter_fig.update_traces(dict(mode="markers"))
        colours = parameter_fig.layout.template.layout.colorway
        for j in range(len(parameter_fig.data)):
            parameter_fig.data[j].name = optim.name
            parameter_fig.data[j].marker.color = colours[i]
            if j > 0:
                parameter_fig.data[j].showlegend = False
        parameter_traces.extend(parameter_fig.data)

        # Collect contour plot data
        contour = pybop.plot.contour(
            result,
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
                    title=dict(text="Cost", font=dict(size=18)),
                    tickfont=dict(size=16),
                ),
                selector=dict(type="contour"),
            )
        else:
            contour.update_traces(showscale=False, selector=dict(type="contour"))

        # Add all traces from the contour figure to the subplot
        for trace in contour.data:
            subplot_contour_fig.add_trace(trace, row=1, col=i + 1)

    # Plot the parameter traces together
    parameter_fig.update_layout(
        width=480,
        height=910,
        plot_bgcolor="white",
        xaxis=dict(title_font_size=16, tickfont_size=16),
        yaxis=dict(
            title=dict(text="Contact resistance / Ω", font_size=16, standoff=30),
            gridcolor=px.colors.qualitative.Pastel2[7],
            zerolinecolor=px.colors.qualitative.Pastel2[7],
            zerolinewidth=1,
            tickfont_size=16,
            range=bounds[0],
        ),
        legend=dict(yanchor="bottom", y=1.02, xanchor="left", x=-0.05, font_size=15),
        xaxis2=dict(title_font_size=16, tickfont_size=16),
        yaxis2=dict(
            title=dict(
                text="Negative particle diffusivity / m<sup>2</sup>&#8239;s<sup>-1</sup>",
                font_size=16,
                standoff=0,
            ),
            gridcolor=px.colors.qualitative.Pastel2[7],
            zerolinecolor=px.colors.qualitative.Pastel2[7],
            zerolinewidth=1,
            tickfont_size=16,
            exponentformat="power",
            range=bounds[1],
        ),
    )
    parameter_fig.data = []
    parameter_fig.add_traces(parameter_traces)
    parameter_fig = plotly.subplots.make_subplots(figure=parameter_fig, rows=2, cols=1)
    parameter_fig.show()
    parameter_fig.write_image("figures/individual/gradient_parameters.pdf")

if create_plot["evolution"]:
    # Create subplot structure
    num_optimisers = len(evolution_optimiser_classes)

    parameter_traces = []
    for i, optimiser in enumerate(evolution_optimiser_classes):
        # Set optimiser options
        if optimiser is pybop.SciPyDifferentialEvolution:
            options = pybop.SciPyDifferentialEvolutionOptions(maxiter=50, popsize=3)
        else:
            options = pybop.PintsOptions(
                max_unchanged_iterations=50, max_evaluations=250
            )

        # Run optimisation
        optim = optimiser(problem, options=options)
        print(optim.name)
        result = optim.run()
        print(result)
        print("True parameter values:", true_value)

        # Plot the parameter traces
        parameter_fig = pybop.plot.parameters(result, show=False, title=None)
        parameter_fig.update_traces(dict(mode="markers"))
        colours = parameter_fig.layout.template.layout.colorway
        for j in range(len(parameter_fig.data)):
            parameter_fig.data[j].name = optim.name
            parameter_fig.data[j].marker.color = colours[i]
            if j > 0:
                parameter_fig.data[j].showlegend = False
        parameter_traces.extend(parameter_fig.data)

        # Collect contour plot data
        contour = pybop.plot.contour(
            result,
            steps=25,
            title="",
            showlegend=False,
            show=False,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        contour.update_traces(showscale=False, selector=dict(type="contour"))
        # Add all traces from the contour figure to the subplot
        for trace in contour.data:
            subplot_contour_fig.add_trace(trace, row=2, col=i + 1)

    # Plot the parameter traces together
    parameter_fig.update_layout(
        width=480,
        height=910,
        plot_bgcolor="white",
        xaxis=dict(title_font_size=16, tickfont_size=16),
        yaxis=dict(
            title=dict(text="Contact resistance / Ω", font_size=16, standoff=30),
            gridcolor=px.colors.qualitative.Pastel2[7],
            zerolinecolor=px.colors.qualitative.Pastel2[7],
            zerolinewidth=1,
            tickfont_size=16,
            range=bounds[0],
        ),
        legend=dict(yanchor="bottom", y=1.02, xanchor="left", x=-0.05, font_size=14),
        xaxis2=dict(title_font_size=16, tickfont_size=16),
        yaxis2=dict(
            title=dict(
                text="Negative particle diffusivity / m<sup>2</sup>&#8239;s<sup>-1</sup>",
                font_size=16,
                standoff=0,
            ),
            gridcolor=px.colors.qualitative.Pastel2[7],
            zerolinecolor=px.colors.qualitative.Pastel2[7],
            zerolinewidth=1,
            tickfont_size=16,
            exponentformat="power",
            range=bounds[1],
        ),
    )
    parameter_fig.data = []
    parameter_fig.add_traces(parameter_traces)
    parameter_fig = plotly.subplots.make_subplots(figure=parameter_fig, rows=2, cols=1)
    parameter_fig.show()
    parameter_fig.write_image("figures/individual/evolution_parameters.pdf")


if create_plot["heuristic"]:
    # Create subplot structure
    num_optimisers = len(evolution_optimiser_classes)

    parameter_traces = []
    for i, optimiser in enumerate(heuristic_optimiser_classes):
        # Set optimiser options
        options = pybop.PintsOptions(max_unchanged_iterations=50, max_evaluations=250)

        # Construct the optimiser
        optim = optimiser(problem, options=options)

        if isinstance(optim, pybop.SimulatedAnnealing):
            optim.optimiser.temperature = 0.1

        # Run optimisation
        print(optim.name)
        result = optim.run()
        print(result)
        print("True parameter values:", true_value)

        # Plot the parameter traces
        parameter_fig = pybop.plot.parameters(result, show=False, title=None)
        parameter_fig.update_traces(dict(mode="markers"))
        colours = parameter_fig.layout.template.layout.colorway
        for j in range(len(parameter_fig.data)):
            parameter_fig.data[j].name = optim.name
            parameter_fig.data[j].marker.color = colours[i]
            if j > 0:
                parameter_fig.data[j].showlegend = False
        parameter_traces.extend(parameter_fig.data)

        # Collect contour plot data
        contour = pybop.plot.contour(
            result,
            steps=25,
            title="",
            showlegend=False,
            show=False,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        contour.update_traces(showscale=False, selector=dict(type="contour"))

        # Add all traces from the contour figure to the subplot
        for trace in contour.data:
            subplot_contour_fig.add_trace(trace, row=3, col=i + 1)

    # Update layout to configure the color bar and plot dimensions
    subplot_contour_fig.update_xaxes(
        dict(title_font_size=12, tickfont_size=16, range=bounds[0])
    )
    subplot_contour_fig.update_yaxes(
        dict(title_font_size=12, tickfont_size=16, range=bounds[1])
    )
    subplot_contour_fig.update_layout(
        showlegend=False,
        height=400 * 3,
        width=400 * max_optims,
    )
    subplot_contour_fig.update_annotations(font_size=18)
    subplot_contour_fig.update_annotations(
        y=-0.01, font_size=18, selector={"text": "Contact resistance / Ω"}
    )

    # Show figure and save image
    subplot_contour_fig.show()
    subplot_contour_fig.write_image("figures/combined/contour_subplot.pdf")

    # Plot the parameter traces together
    parameter_fig.update_layout(
        width=480,
        height=910,
        plot_bgcolor="white",
        xaxis=dict(title_font_size=16, tickfont_size=16),
        yaxis=dict(
            title=dict(text="Contact resistance / Ω", font_size=16, standoff=30),
            gridcolor=px.colors.qualitative.Pastel2[7],
            zerolinecolor=px.colors.qualitative.Pastel2[7],
            zerolinewidth=1,
            tickfont_size=16,
            range=bounds[0],
        ),
        legend=dict(yanchor="bottom", y=1.02, xanchor="left", x=-0.05, font_size=15),
        xaxis2=dict(title_font_size=16, tickfont_size=16),
        yaxis2=dict(
            title=dict(
                text="Negative particle diffusivity / m<sup>2</sup>&#8239;s<sup>-1</sup>",
                font_size=16,
                standoff=0,
            ),
            gridcolor=px.colors.qualitative.Pastel2[7],
            zerolinecolor=px.colors.qualitative.Pastel2[7],
            zerolinewidth=1,
            tickfont_size=16,
            exponentformat="power",
            range=bounds[1],
        ),
    )
    parameter_fig.data = []
    parameter_fig.add_traces(parameter_traces)
    parameter_fig = plotly.subplots.make_subplots(figure=parameter_fig, rows=2, cols=1)
    parameter_fig.show()
    parameter_fig.write_image("figures/individual/heuristic_parameters.pdf")


if create_plot["posteriors"]:
    sigma0 = pybop.Parameter(
        initial_value=sigma,
        prior=pybop.Uniform(1e-8 * sigma, 10 * sigma),
        bounds=[1e-8, 10 * sigma],
    )
    likelihood = pybop.GaussianLogLikelihood(dataset, sigma0=sigma0)
    posterior = pybop.Problem(simulator, pybop.LogPosterior(likelihood))

    options = pybop.PintsSamplerOptions(
        n_chains=5,
        max_iterations=3500,
        warm_up_iterations=1500,
        cov=posterior.parameters.get_sigma0(transformed=True),
    )
    sampler = pybop.HaarioBardenetACMC(posterior, options=options)

    result = sampler.run()
    print(result)
    print("True parameter values:", [true_value, sigma])
    summary = pybop.PosteriorSummary(result.chains)
    print(summary.rhat())
    print(summary.effective_sample_size(mixed_chains=True))

    # Create a grid for subplots
    fig = plt.figure(figsize=(15, 6))
    gs = fig.add_gridspec(1, 3, height_ratios=[1])

    # Function to format axis
    def format_axis(ax):
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-1, 1))  # Set limits to use scientific notation
        formatter.set_scientific(True)  # Ensure scientific notation
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
        ax.tick_params(axis="both", which="major", labelsize=14)

    # Subplot for parameter 0
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(
        summary.all_samples[:, 0],
        bins=50,
        density=False,
        alpha=0.6,
        color="tab:blue",
    )
    ax1.set_xlabel(r"Contact resistance / $\Omega$", fontsize=16)
    ax1.set_ylabel("Density", fontsize=16)
    ax1.set_ylim(0, 775)
    ax1.tick_params(axis="both", which="major", labelsize=16)
    ax1.axvspan(
        summary.get_summary_statistics()[("ci_lower")][0],
        summary.get_summary_statistics()[("ci_upper")][0],
        alpha=0.1,
        color="tab:blue",
    )
    ax1.axvline(x=true_value[0], color="black")
    format_axis(ax1)

    # Subplot for parameter 1
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(
        summary.all_samples[:, 1],
        bins=50,
        density=False,
        alpha=0.6,
        color="tab:red",
        label="$R_n$",
    )
    ax2.set_xlabel(r"Negative particle diffusivity / m$^2\,$s$^{-1}$", fontsize=16)
    ax2.set_ylim(0, 775)
    ax2.tick_params(axis="both", which="major", labelsize=16)
    ax2.axvspan(
        summary.get_summary_statistics()[("ci_lower")][1],
        summary.get_summary_statistics()[("ci_upper")][1],
        alpha=0.1,
        color="tab:red",
    )
    ax2.axvline(x=true_value[1], color="black")
    format_axis(ax2)

    # Subplot for sigma
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(
        summary.all_samples[:, 2],
        bins=50,
        density=False,
        alpha=0.6,
        color="tab:purple",
    )
    ax3.set_xlabel("Observation Noise / V", fontsize=16)
    ax3.set_ylim(0, 775)
    ax3.tick_params(axis="both", which="major", labelsize=16)
    ax3.axvspan(
        summary.get_summary_statistics()[("ci_lower")][2],
        summary.get_summary_statistics()[("ci_upper")][2],
        alpha=0.1,
        color="tab:purple",
    )
    ax3.axvline(x=sigma, color="black")
    format_axis(ax3)

    # Adjust layout
    plt.tight_layout()
    plt.savefig("figures/combined/posteriors.pdf")
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
    model = pybamm.lithium_ion.SPM(
        options={"surface form": "differential", "contact resistance": "true"}
    )

    initial_state = {"Initial SoC": 0.05}
    n_frequency = 35
    sigma0 = 5e-4
    f_eval = np.logspace(-3.5, 4, n_frequency)

    # Create synthetic data for parameter inference
    simulator = pybop.pybamm.EISSimulator(
        model,
        parameter_values=parameter_values,
        f_eval=f_eval,
        initial_state=initial_state,
        var_pts=var_pts,
    )
    solution = simulator.solve(
        inputs={
            "Contact resistance [Ohm]": true_value[0],
            "Negative particle diffusivity [m2.s-1]": true_value[1],
        }
    )

    # Form dataset
    dataset = pybop.Dataset(
        {
            "Frequency [Hz]": f_eval,
            "Current function [A]": np.zeros(len(f_eval)),
            "Impedance": solution["Impedance"].data
            + noise(sigma0, len(solution["Impedance"].data)),
        },
        domain="Frequency [Hz]",
    )

    # Build the problem
    cost = pybop.RootMeanSquaredError(dataset, target=["Impedance"])
    problem = pybop.Problem(simulator, cost)

    # Set up and run the optimiser
    options = pybop.PintsOptions(max_iterations=75, min_iterations=75)
    optim = pybop.CMAES(problem, options=options)
    result = optim.run()

    parameter_fig = pybop.plot.nyquist(
        problem,
        result.best_inputs,
        title="",
        width=595,
        height=595,
        margin=dict(r=50, t=50),
        xaxis_title_font_size=16,
        yaxis_title_font_size=16,
    )
    parameter_fig[0].data[1].update(line=dict(color="#00CC97"))
    parameter_fig[0].write_image("figures/individual/impedance_spectrum.pdf")

    landscape_fig = pybop.plot.contour(
        problem,
        steps=25,
        show=False,
        xaxis=dict(
            title=dict(text="Contact resistance / Ω", font_size=16), tickfont_size=15
        ),
        yaxis=dict(
            title=dict(
                text="Negative particle diffusivity / m<sup>2</sup>&#8239;s<sup>-1</sup>",
                font_size=16,
                standoff=0,
            ),
            tickfont_size=15,
            exponentformat="power",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=15),
        ),
        coloraxis_colorbar=dict(tickfont=dict(size=18)),
        margin=dict(t=50),
        title=None,
    )
    landscape_fig.add_trace(
        go.Scatter(
            x=[initial_value[0]],
            y=[initial_value[1]],
            mode="markers",
            marker=dict(
                symbol="x",
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
            marker=dict(
                symbol="cross",
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
    landscape_fig.write_image("figures/individual/impedance_contour.pdf")
