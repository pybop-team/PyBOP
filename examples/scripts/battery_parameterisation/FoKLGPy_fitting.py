"""
Concentration-dependent electrolyte diffusivity identified using a FoKL
decomposed Gaussian process (GP).

FoKL (Forward variable selection using Karhunen-Loève decomposed Gaussian
Processes) represents an unknown function as a weighted sum of basis functions
(here, Bernoulli polynomials) of the normalised inputs. The weights ("Beta"
parameters) become standard PyBOP parameters with Gaussian priors, so the
function can be fitted with any PyBOP optimiser. With "exp" set to True, the
GP models log(D_e), which guarantees D_e(c_e) > 0.

In this example, synthetic data is generated from a DFN model with the Chen2020
parameter set. The electrolyte diffusivity D_e(c_e) is fitted to a 2C discharge,
once as a FoKL GP and once as a constant value. Both fits are then validated on
a 3C discharge, and the identified diffusivities are compared with the ground
truth.

Reference:
    K. Hayes, M.W. Fouts, A. Baheri and D.S. Mebane, "Forward variable selection
    enables fast and accurate dynamic system identification with Karhunen-Loève
    decomposed Gaussian processes", arXiv:2205.13676.
    https://arxiv.org/abs/2205.13676

FoKL package: https://pypi.org/project/FoKL/
"""

import numpy as np
import plotly.graph_objects as go
import pybamm

import pybop

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
        xaxis_title="Time / s",
        yaxis_title="Voltage / V",
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

fitted = {}
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
    fitted[name] = param

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

# Compare the identified electrolyte diffusivity with the ground truth
c_e = np.linspace(1, 3000, 200)  # matches the GP normalisation range


def D_e(parameter_values):
    return parameter_values.evaluate(
        pybamm.FunctionParameter(
            "Electrolyte diffusivity [m2.s-1]",
            {
                "Electrolyte concentration [mol.m-3]": pybamm.Vector(c_e),
                "Temperature [K]": pybamm.Scalar(298.15),
            },
        )
    ).flatten()


fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=c_e,
        y=D_e(parameter_values),
        mode="lines",
        name="Ground truth (Chen2020)",
        line=dict(color="black", dash="dash"),
    )
)
for (name, param), colour in zip(fitted.items(), ["green", "purple"], strict=True):
    fig.add_trace(
        go.Scatter(
            x=c_e,
            y=D_e(param),
            mode="lines",
            name=name,
            line=dict(color=colour),
        )
    )
fig.update_layout(
    xaxis_title="Electrolyte concentration / mol.m-3",
    yaxis_title="Electrolyte diffusivity / m2.s-1",
    title="Identified electrolyte diffusivity",
)
figures.append(fig)

for fig in figures:
    fig.show()
