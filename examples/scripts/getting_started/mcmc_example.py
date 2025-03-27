import sys

import numpy as np
import pybamm

import pybop

# Set parallelization if on macOS / Unix
parallel = True if sys.platform != "win32" else False

# Parameter set and model definition
solver = pybamm.IDAKLUSolver()
parameter_set = pybop.ParameterSet("Chen2020")
parameter_set.update(
    {
        "Negative electrode active material volume fraction": 0.63,
        "Positive electrode active material volume fraction": 0.71,
    }
)
synth_model = pybop.lithium_ion.DFN(parameter_set=parameter_set, solver=solver)

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

# Generate data
init_soc = 0.5
sigma = 0.005
experiment = pybop.Experiment(
    [
        ("Discharge at 0.5C for 6 minutes (5 second period)",),
    ]
)
values = synth_model.predict(
    initial_state={"Initial SoC": init_soc}, experiment=experiment
)


def noisy(data, sigma):
    return data + np.random.normal(0, sigma, len(data))


# Form dataset
dataset = pybop.Dataset(
    {
        "Time [s]": values["Time [s]"].data,
        "Current function [A]": values["Current [A]"].data,
        "Voltage [V]": noisy(values["Voltage [V]"].data, sigma),
        "Bulk open-circuit voltage [V]": noisy(
            values["Bulk open-circuit voltage [V]"].data, sigma
        ),
    }
)

model = pybop.lithium_ion.SPM(parameter_set=parameter_set, solver=pybamm.IDAKLUSolver())
signal = ["Voltage [V]", "Bulk open-circuit voltage [V]"]

# Generate problem, likelihood, and sampler
problem = pybop.FittingProblem(
    model, parameters, dataset, signal=signal, initial_state={"Initial SoC": init_soc}
)
likelihood = pybop.GaussianLogLikelihood(problem)
posterior = pybop.LogPosterior(likelihood)

sampler = pybop.DifferentialEvolutionMCMC(
    posterior,
    chains=3,
    max_iterations=300,
    warm_up=100,
    verbose=True,
    parallel=parallel,  # (macOS/WSL/Linux only)
)
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
