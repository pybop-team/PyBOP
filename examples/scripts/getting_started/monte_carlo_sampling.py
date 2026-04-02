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
dataset = pybop.Dataset(
    {
        "Time [s]": solution.t,
        "Current [A]": solution["Current [A]"].data,
        "Voltage [V]": pybop.add_noise(solution["Voltage [V]"].data, sigma),
    }
)

# Fitting parameters
parameter_values.update(
    {
        "Negative electrode active material volume fraction": pybop.Parameter(
            distribution=pybop.LogNormal(np.log(0.68), 0.02),
            transformation=pybop.LogTransformation(),
        ),
        "Positive electrode active material volume fraction": pybop.Parameter(
            distribution=pybop.LogNormal(np.log(0.65), 0.02),
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
cost = pybop.GaussianLogLikelihood(dataset)
log_pdf = pybop.LogPosterior(simulator, cost)

# Create and run the sampler
options = pybop.PintsSamplerOptions(
    n_chains=3,
    max_iterations=250,  # Extend this for accurate posteriors
    warm_up_iterations=100,
    verbose=True,
)
sampler = pybop.DifferentialEvolutionMCMC(log_pdf, options=options)
result = sampler.run()

# Summary statistics
summary = result.get_summary_statistics()
print(summary)
result.plot_trace()
result.summary_table()
result.plot_posterior()
result.plot_chains()
result.effective_sample_size()
print(f"rhat: {result.rhat()}")
