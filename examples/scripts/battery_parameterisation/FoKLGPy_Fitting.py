import numpy as np
import plotly.graph_objects as go
import pybamm

import pybop

"""
In this example, we demonstrate the use of FoKL decomposed Gaussian Processes
for state-dependent estimation of electrolyte diffusivity.
"""

# Define model and parameter values
model = pybamm.lithium_ion.DFN()
parameter_values = pybamm.ParameterValues("Chen2020")

# Generate synthetic training and validation datasets and add some white noise
experiments = [
    pybamm.Experiment(["Discharge at 2C until 2.5 V"]),
    pybamm.Experiment(["Discharge at 3C until 2.7 V"]),
]

sigma = 5e-3
datasets = []
figures = []
for experiment, title in zip(
    experiments,
    ["Training experiment (2C discharge)", "Validation experiment (3C discharge)"],
    strict=True,
):
    solution = pybamm.Simulation(
        model, experiment=experiment, parameter_values=parameter_values
    ).solve()

    dataset = pybop.import_pybamm_solution(solution)
    dataset["Voltage [V]"] = pybop.add_noise(dataset["Voltage [V]"], sigma)
    datasets.append(dataset)

    # Set up figures
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dataset["Time [s]"],
            y=dataset["Voltage [V]"],
            mode="markers",
            name="Data",
            marker=dict(size=4),
        )
    )
    fig.update_layout(
        xaxis_title="Time [s]",
        yaxis_title="Voltage [V]",
        title=title,
        legend=dict(x=0.01, y=0.01),
    )
    figures.append(fig)

# Configure FoKL GP parameter representation, using exp=True guarantees that
# D_e(c_e) > 0 during optimisation
GP_options = {
    "Number of terms": 3,
    "Arguments": ["Electrolyte concentration [mol.m-3]"],
    "Normalization min-max": {"Electrolyte concentration [mol.m-3]": (0.0, 3000.0)},
    "Constant mean": np.log(1.8e-10),
    "Constant standard deviation": 0.2,
    "Bi mean": 0.0,
    "Bi standard deviation": 0.2,
    "exp": True,
    "Verbose": True,
}

GP_param = pybop.FoKLGP(
    "Electrolyte diffusivity [m2.s-1]",
    parameter_values=parameter_values.copy(),
    options=GP_options,
    twoway=True,
    model=model,
)
parameters_gp = GP_param.get_parameter_values()

# Configure constant scalar parameter representation for comparison
parameters_constant = parameter_values.copy()
parameters_constant.update(
    {
        "Electrolyte diffusivity [m2.s-1]": pybop.Parameter(
            pybop.Gaussian(1.79e-10, 2e-11, truncated_at=[1e-14, 1e-7])
        )
    }
)

# Build shared cost and optimiser options
training_data = datasets[0]
cost = pybop.GaussianLogLikelihoodKnownSigma(training_data, sigma=sigma)
options = pybop.PintsOptions(max_iterations=250, max_unchanged_iterations=50)

for name, param, colour in zip(
    ["FoKL GP", "PyBOP Constant"],
    [parameters_gp, parameters_constant],
    ["green", "purple"],
    strict=True,
):
    # Build simulator and problem
    simulator = pybop.pybamm.Simulator(
        model, parameter_values=param, protocol=training_data
    )
    problem = pybop.Problem(simulator, cost)

    # Run optimisation
    optim = pybop.XNES(problem, options=options)
    result = optim.run()
    print(name, result)

    # Plot optimisation diagnostics
    if name == "FoKL GP":
        result.plot_convergence()
        result.plot_parameters()

    # Update the parameter values
    param.update(result.best_inputs)

    # Evaluate optimised parameters on Training (2C) and Validation (3C) experiments
    for experiment, fig in zip(experiments, figures, strict=True):
        solution = pybamm.Simulation(
            model, experiment=experiment, parameter_values=param
        ).solve()

        # Plot for comparison
        fig.add_trace(
            go.Scatter(
                x=solution["Time [s]"].entries,
                y=solution["Voltage [V]"].entries,
                mode="lines",
                name=name,
                line=dict(color=colour),
            )
        )

for fig in figures:
    fig.show()
