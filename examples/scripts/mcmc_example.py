import numpy as np
import plotly.graph_objects as go

import pybop

# Parameter set and model definition
parameter_set = pybop.ParameterSet.pybamm("Chen2020")
model = pybop.lithium_ion.SPM(parameter_set=parameter_set)

# Fitting parameters
parameters = [
    pybop.Parameter(
        "Negative electrode active material volume fraction",
        prior=pybop.Gaussian(0.68, 0.05),
    ),
    pybop.Parameter(
        "Positive electrode active material volume fraction",
        prior=pybop.Gaussian(0.58, 0.05),
    ),
]

# Generate data
init_soc = 0.5
sigma = 0.001
experiment = pybop.Experiment(
    [
        (
            "Discharge at 0.5C for 3 minutes (2 second period)",
            "Charge at 0.5C for 3 minutes (2 second period)",
        ),
    ]
    * 2
)
values = model.predict(init_soc=init_soc, experiment=experiment)


def noise(sigma):
    return np.random.normal(0, sigma, len(values["Voltage [V]"].data))


# Form dataset
dataset = pybop.Dataset(
    {
        "Time [s]": values["Time [s]"].data,
        "Current function [A]": values["Current [A]"].data,
        "Voltage [V]": values["Voltage [V]"].data + noise(sigma),
        "Bulk open-circuit voltage [V]": values["Bulk open-circuit voltage [V]"].data
        + noise(sigma),
    }
)

signal = ["Voltage [V]", "Bulk open-circuit voltage [V]"]

# Generate problem, cost function, and optimisation class
problem = pybop.FittingProblem(
    model, parameters, dataset, signal=signal, init_soc=init_soc
)
likelihood = pybop.GaussianLogLikelihoodKnownSigma(problem, sigma=[0.02, 0.02])
prior1 = pybop.Gaussian(0.7, 0.02)
prior2 = pybop.Gaussian(0.6, 0.02)
composed_prior = pybop.ComposedLogPrior(prior1, prior2)
posterior = pybop.LogPosterior(likelihood, composed_prior)
x0 = [[0.68, 0.58], [0.68, 0.58], [0.68, 0.58]]

optim = pybop.DREAM(
    posterior,
    chains=3,
    x0=x0,
    max_iterations=400,
    initial_phase_iterations=250,
    # parallel=True, # uncomment to enable parallelisation (MacOS/Linux only)
)
result = optim.run()


# Create a histogram
fig = go.Figure()
for i, data in enumerate(result):
    fig.add_trace(go.Histogram(x=data[:, 0], name="Neg", opacity=0.75))
    fig.add_trace(go.Histogram(x=data[:, 1], name="Pos", opacity=0.75))

# Update layout for better visualization
fig.update_layout(
    title="Posterior distribution of volume fractions",
    xaxis_title="Value",
    yaxis_title="Count",
    barmode="overlay",
)

# Show the plot
fig.show()
