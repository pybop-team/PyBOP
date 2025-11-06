import numpy as np
import pybamm

import pybop

# Define model and parameter values
model = pybamm.lithium_ion.SPM(
    options={"surface form": "differential", "contact resistance": "true"},
)
parameter_values = pybamm.ParameterValues("Chen2020")
parameter_values["Contact resistance [Ohm]"] = 0.0
parameter_values.update(
    {
        "Negative electrode active material volume fraction": 0.531,
        "Positive electrode active material volume fraction": 0.732,
    }
)
parameter_values.set_initial_state(0.5)
n_frequency = 20
sigma0 = 1e-4
f_eval = np.logspace(-4, 5, n_frequency)

# Create synthetic data for parameter inference
solution = pybop.pybamm.EISSimulator(
    model, parameter_values=parameter_values, f_eval=f_eval
).simulate()


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
        "Current function [A]": np.zeros_like(f_eval),
        "Impedance": noisy(solution["Impedance"], sigma0),
    },
    domain="Frequency [Hz]",
)

# Fitting parameters
parameter_values.update(
    {
        "Negative electrode active material volume fraction": pybop.Uniform(0.4, 0.75),
        "Positive electrode active material volume fraction": pybop.Uniform(0.4, 0.75),
    }
)

# Build the problem
simulator = pybop.pybamm.EISSimulator(
    model, parameter_values=parameter_values, f_eval=dataset["Frequency [Hz]"]
)
cost = pybop.GaussianLogLikelihoodKnownSigma(dataset, target="Impedance", sigma0=sigma0)
problem = pybop.Problem(simulator, cost)

# Set up the optimiser
options = pybop.PintsOptions(
    max_iterations=100, sigma=0.25, max_unchanged_iterations=30
)
optim = pybop.CMAES(problem, options=options)

# Run the optimisation
result = optim.run()
print(result)

# Plot the nyquist
pybop.plot.nyquist(problem, problem_inputs=result.x, title="Optimised Comparison")

# Plot the optimisation result
result.plot_convergence()
result.plot_parameters()
result.plot_surface()
