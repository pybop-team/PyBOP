import numpy as np
import pybamm

import pybop

# Parameter set and model definition
solver = pybamm.IDAKLUSolver()
parameter_set = pybop.ParameterSet.pybamm("Chen2020")
parameter_set.update(
    {
        "Negative electrode active material volume fraction": 0.66,
        "Positive electrode active material volume fraction": 0.68,
    }
)
synth_model = pybop.lithium_ion.DFN(parameter_set=parameter_set, solver=solver)

# Fitting parameters
parameters = pybop.Parameters(
    pybop.Parameter(
        "Negative electrode active material volume fraction",
        prior=pybop.Gaussian(0.65, 0.05),
    ),
    pybop.Parameter(
        "Positive electrode active material volume fraction",
        prior=pybop.Gaussian(0.65, 0.05),
    ),
)

# Generate data
init_soc = 0.5
sigma = 0.002


def noise(sigma):
    return np.random.normal(0, sigma, len(values["Voltage [V]"].data))


experiment = pybop.Experiment(
    [
        (
            "Discharge at 0.5C for 1 minutes (5 second period)",
            "Charge at 0.5C for 1 minutes (5 second period)",
            "Discharge at 3C for 20 seconds (1 second period)",
        ),
    ]
)
values = synth_model.predict(
    initial_state={"Initial SoC": init_soc}, experiment=experiment
)

# Form dataset
dataset = pybop.Dataset(
    {
        "Time [s]": values["Time [s]"].data,
        "Current function [A]": values["Current [A]"].data,
        "Voltage [V]": values["Voltage [V]"].data + noise(sigma),
    }
)

# Generate problem, likelihood, and sampler
model = pybop.lithium_ion.DFN(parameter_set=parameter_set, solver=pybamm.IDAKLUSolver())
model.build(initial_state={"Initial SoC": init_soc})
problem = pybop.FittingProblem(model, parameters, dataset)
likelihood = pybop.GaussianLogLikelihoodKnownSigma(problem, sigma0=sigma)
prior = pybop.JointLogPrior(*parameters.priors())

sampler = pybop.AnnealedImportanceSampler(
    likelihood, prior, iterations=10, num_beta=300, cov0=np.eye(2) * 1e-2
)
mean, median, std, var = sampler.run()

print(f"mean: {mean}, std: {std}, median: {median}, var: {var}")
