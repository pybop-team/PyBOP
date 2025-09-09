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

# Generate a synthetic dataset
sim = pybamm.Simulation(model, parameter_values=parameter_values)
t_eval = np.linspace(0, 1000, 240)
sol = sim.solve(t_eval=t_eval)

sigma = 5e-3
dataset = pybop.Dataset(
    {
        "Time [s]": t_eval,
        "Voltage [V]": sol["Voltage [V]"](t_eval),
        "Current function [A]": sol["Current [A]"](t_eval)
        + np.random.normal(0, sigma, len(t_eval)),
    }
)

# Construct the parameters to be sampled
parameters = [
    pybop.Parameter(
        "Negative electrode active material volume fraction",
        prior=pybop.Gaussian(0.6, 0.2),
        bounds=[0.5, 0.8],
    ),
    pybop.Parameter(
        "Positive electrode active material volume fraction",
        prior=pybop.Gaussian(0.6, 0.2),
        bounds=[0.5, 0.8],
    ),
]

# Create the problem builder
builder = (
    pybop.builders.Pybamm()
    .set_dataset(dataset)
    .set_simulation(model, parameter_values=parameter_values)
    # We leave sigma non-defined in the below cost as we want to estimate it alongside the parameters
    .add_cost(pybop.costs.pybamm.NegativeGaussianLogLikelihood("Voltage [V]"))
)
for param in parameters:
    builder.add_parameter(param)
problem = builder.build()

# Set the sampler options and construct the sampler
options = pybop.PintsSamplerOptions(
    n_chains=3,
    warm_up_iterations=50,  # Extend this for accurate posteriors
    max_iterations=300,  # Extend this for accurate posteriors
    verbose=True,
    cov=1e-2,
)
sampler = pybop.DifferentialEvolutionMCMC(problem, options=options)
chains = sampler.run()

# Summary statistics and plotting
posterior_summary = pybop.PosteriorSummary(chains)
print(posterior_summary.get_summary_statistics())
posterior_summary.plot_trace()
posterior_summary.summary_table()
posterior_summary.plot_chains()
posterior_summary.plot_posterior()
posterior_summary.effective_sample_size()
print(f"rhat: {posterior_summary.rhat()}")
