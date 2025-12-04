import numpy as np
import pybamm

import pybop

"""
In this example, we present a PyBOP's Monte Carlo Sampler framework. Monte Carlo
sampling provides a method to resolve intractable integration problems. In PyBOP,
we use this to integrate Bayes formula providing uncertainty insights via the
sampled posterior.
"""

# Set the model and parameter values
model = pybamm.lithium_ion.SPM()
parameter_values = pybamm.ParameterValues("Chen2020")
parameter_values.set_initial_state(0.5)

# Generate a synthetic dataset
sigma = 0.005
experiment = pybamm.Experiment(["Discharge at 0.5C for 3 minutes (5 second period)"])
solution = pybamm.Simulation(
    model, parameter_values=parameter_values, experiment=experiment
).solve()


def noisy(data, sigma):
    return data + np.random.normal(0, sigma, len(data))


dataset = pybop.Dataset(
    {
        "Time [s]": solution.t,
        "Current function [A]": solution["Current [A]"].data,
        "Voltage [V]": noisy(solution["Voltage [V]"].data, sigma),
    }
)

# Fitting parameters
parameter_values.update(
    {
        "Negative electrode active material volume fraction": pybop.Parameter(
            distribution=pybop.Gaussian(0.68, 0.02),
            transformation=pybop.LogTransformation(),
        ),
        "Positive electrode active material volume fraction": pybop.Parameter(
            distribution=pybop.Gaussian(0.65, 0.02),
            transformation=pybop.LogTransformation(),
        ),
    }
)

# Define model (and use existing parameter values)
model = pybamm.lithium_ion.SPM()

# Build the problem
simulator = pybop.pybamm.Simulator(
    model, parameter_values=parameter_values, protocol=dataset
)
likelihood = pybop.GaussianLogLikelihood(dataset)
posterior = pybop.LogPosterior(likelihood)
problem = pybop.Problem(simulator, posterior)

# Create and run the sampler
options = pybop.PintsSamplerOptions(
    n_chains=3,
    max_iterations=250,  # Extend this for accurate posteriors
    warm_up_iterations=100,
    verbose=True,
)
sampler = pybop.DifferentialEvolutionMCMC(problem, options=options)
result = sampler.run()

# Summary statistics
posterior_summary = pybop.PosteriorSummary(result.chains)
print(posterior_summary.get_summary_statistics())
posterior_summary.plot_trace()
posterior_summary.summary_table()
posterior_summary.plot_posterior()
posterior_summary.plot_chains()
posterior_summary.effective_sample_size()
print(f"rhat: {posterior_summary.rhat()}")
