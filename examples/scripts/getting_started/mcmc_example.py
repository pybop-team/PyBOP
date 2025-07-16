import numpy as np
import pybamm

import pybop

model = pybamm.lithium_ion.SPM()
parameter_values = pybamm.ParameterValues("Chen2020")
t_eval = np.linspace(0, 100, 240)
sim = pybamm.Simulation(
    model=model,
    parameter_values=parameter_values,
)
sol = sim.solve(t_eval=t_eval)

dataset = pybop.Dataset(
    {
        "Time [s]": sol.t,
        "Voltage [V]": sol["Voltage [V]"].data,
        "Current function [A]": sol["Current [A]"].data,
    }
)

# Create the builder
builder = pybop.builders.Pybamm()
builder.set_dataset(dataset)
builder.set_simulation(
    model,
    parameter_values=parameter_values,
)
builder.add_parameter(
    pybop.Parameter(
        "Negative electrode active material volume fraction",
        initial_value=0.6,
        prior=pybop.Gaussian(0.6, 0.2),
        transformation=pybop.LogTransformation(),
        bounds=[0.5, 0.8],
    )
)
builder.add_parameter(
    pybop.Parameter(
        "Positive electrode active material volume fraction",
        initial_value=0.6,
        prior=pybop.Gaussian(0.6, 0.2),
        transformation=pybop.LogTransformation(),
        bounds=[0.5, 0.8],
    )
)

builder.add_cost(
    pybop.costs.pybamm.NegativeGaussianLogLikelihood("Voltage [V]", "Voltage [V]")
)
# Build the problem
problem = builder.build()

options = pybop.PintsSamplerOptions(
    n_chains=2,
    warm_up_iterations=50,
    max_iterations=400,
    verbose=True,
    cov=1e-2,
)
sampler = pybop.AdaptiveCovarianceMCMC(problem, options=options)
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
