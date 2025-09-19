import sys

import numpy as np
import pybamm

import pybop

# Set parallelization if on macOS / Unix
parallel = True if sys.platform != "win32" else False

# Define model and parameter values
synth_model = pybamm.lithium_ion.SPMe()
parameter_values = pybamm.ParameterValues("Chen2020")
parameter_values.update(
    {
        "Negative electrode active material volume fraction": 0.63,
        "Positive electrode active material volume fraction": 0.71,
    }
)
parameter_values.set_initial_state(0.5)

# Fitting parameters
parameters = pybop.Parameters(
    pybop.Parameter(
        "Negative electrode active material volume fraction",
        prior=pybop.Gaussian(0.68, 0.02),
        transformation=pybop.LogTransformation(),
    ),
    pybop.Parameter(
        "Positive electrode active material volume fraction",
        prior=pybop.Gaussian(0.65, 0.02),
        transformation=pybop.LogTransformation(),
    ),
)

# Generate a synthetic dataset
sigma = 0.005
experiment = pybamm.Experiment(["Discharge at 0.5C for 3 minutes (5 second period)"])
sim = pybamm.Simulation(
    synth_model, parameter_values=parameter_values, experiment=experiment
)
sol = sim.solve()


def noisy(data, sigma):
    return data + np.random.normal(0, sigma, len(data))


dataset = pybop.Dataset(
    {
        "Time [s]": sol.t,
        "Current function [A]": sol["Current [A]"].data,
        "Voltage [V]": noisy(sol["Voltage [V]"].data, sigma),
    }
)

# Define model (and use existing parameter values)
model = pybamm.lithium_ion.SPM()

# Build the problem
simulator = pybop.pybamm.Simulator(
    model,
    parameter_values=parameter_values,
    input_parameter_names=parameters.names,
    protocol=dataset,
)
problem = pybop.FittingProblem(simulator, parameters, dataset)
likelihood = pybop.GaussianLogLikelihood(problem)
posterior = pybop.LogPosterior(likelihood)

# Create and run the sampler
options = pybop.PintsSamplerOptions(
    n_chains=3,
    max_iterations=250,  # Reduced for CI, increase for improved posteriors
    warm_up_iterations=100,
    verbose=True,
)
sampler = pybop.DifferentialEvolutionMCMC(posterior, options=options)
chains = sampler.run()

# Summary statistics
posterior_summary = pybop.PosteriorSummary(chains)
print(posterior_summary.get_summary_statistics())
posterior_summary.plot_trace()
posterior_summary.summary_table()
posterior_summary.plot_posterior()
posterior_summary.plot_chains()
posterior_summary.effective_sample_size()
print(f"rhat: {posterior_summary.rhat()}")
