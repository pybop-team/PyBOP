import numpy as np

import pybop

# Parameter set and model definition
parameter_set = pybop.ParameterSet.pybamm("Chen2020")
synth_model = pybop.lithium_ion.DFN(parameter_set=parameter_set)

# Fitting parameters
parameters = [
    pybop.Parameter(
        "Negative electrode thickness [m]",
        prior=pybop.Gaussian(8e-05, 0.5e-05),
        true_value=parameter_set["Negative electrode thickness [m]"],
    ),
    pybop.Parameter(
        "Positive electrode thickness [m]",
        prior=pybop.Gaussian(7.5e-05, 0.5e-05),
        true_value=parameter_set["Positive electrode thickness [m]"],
    ),
]

# Generate data
init_soc = 1.0
sigma = 0.001
experiment = pybop.Experiment(
    [
        (
            "Discharge at 0.5C until 2.5V (10 second period)",
            "Charge at 0.5C until 4.2V (10 second period)",
        ),
    ]
    # * 2
)
values = synth_model.predict(init_soc=init_soc, experiment=experiment)


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

model = pybop.lithium_ion.SPMe(parameter_set=parameter_set)
signal = ["Voltage [V]", "Bulk open-circuit voltage [V]"]

# Generate problem, likelihood, and sampler
problem = pybop.FittingProblem(
    model, parameters, dataset, signal=signal, init_soc=init_soc
)
likelihood = pybop.GaussianLogLikelihoodKnownSigma(problem, sigma0=0.002)
prior1 = pybop.Gaussian(8e-05, 0.5e-05)
prior2 = pybop.Gaussian(7.5e-05, 0.5e-05)
composed_prior = pybop.ComposedLogPrior(prior1, prior2)
posterior = pybop.LogPosterior(likelihood, composed_prior)

x0 = []
n_chains = 10
for _i in range(n_chains):
    x0.append(np.array([8.5e-05, 7.5e-05]))

optim = pybop.DREAM(
    posterior,
    chains=n_chains,
    x0=x0,
    max_iterations=300,
    burn_in=100,
    parallel=True,
)
result = optim.run()

# Summary statistics
posterior_summary = pybop.PosteriorSummary(result)
print(posterior_summary.get_summary_statistics())
posterior_summary.plot_trace()
posterior_summary.summary_table()
posterior_summary.plot_posterior()
