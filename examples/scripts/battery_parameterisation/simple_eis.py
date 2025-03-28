import numpy as np

import pybop

# Define model
parameter_set = pybop.ParameterSet("Chen2020")
parameter_set["Contact resistance [Ohm]"] = 0.0
initial_state = {"Initial SoC": 0.5}
n_frequency = 20
sigma0 = 1e-4
f_eval = np.logspace(-4, 5, n_frequency)
model = pybop.lithium_ion.SPM(
    parameter_set=parameter_set,
    eis=True,
    options={"surface form": "differential", "contact resistance": "true"},
)

# Create synthetic data for parameter inference
sim = model.simulateEIS(
    inputs={
        "Negative electrode active material volume fraction": 0.531,
        "Positive electrode active material volume fraction": 0.732,
    },
    f_eval=f_eval,
    initial_state=initial_state,
)

# Fitting parameters
parameters = pybop.Parameters(
    pybop.Parameter(
        "Negative electrode active material volume fraction",
        prior=pybop.Uniform(0.4, 0.75),
        bounds=[0.375, 0.75],
    ),
    pybop.Parameter(
        "Positive electrode active material volume fraction",
        prior=pybop.Uniform(0.4, 0.75),
        bounds=[0.375, 0.75],
    ),
)


def noisy(data, sigma):
    # Generate real part noise
    real_noise = np.random.normal(0, sigma, len(data))

    # Generate imaginary part noise
    imag_noise = np.random.normal(0, sigma, len(data))

    # Combine them into a complex noise
    return data + real_noise + 1j * imag_noise


# Form dataset
dataset = pybop.Dataset(
    {
        "Frequency [Hz]": f_eval,
        "Current function [A]": np.ones(n_frequency) * 0.0,
        "Impedance": noisy(sim["Impedance"], sigma0),
    }
)

signal = ["Impedance"]
# Generate problem, cost function, and optimisation class
problem = pybop.FittingProblem(model, parameters, dataset, signal=signal)
cost = pybop.GaussianLogLikelihoodKnownSigma(problem, sigma0=sigma0)
optim = pybop.CMAES(cost, max_iterations=100, sigma0=0.25, max_unchanged_iterations=30)

results = optim.run()

# Plot the nyquist
pybop.plot.nyquist(problem, problem_inputs=results.x, title="Optimised Comparison")

# Plot convergence
pybop.plot.convergence(optim)

# Plot the parameter traces
pybop.plot.parameters(optim)

# Plot 2d landscape
pybop.plot.surface(optim)
