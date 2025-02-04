import numpy as np

import pybop

# Define model and use high-performant solver for sensitivities
parameter_set = pybop.ParameterSet("Chen2020")
model = pybop.lithium_ion.SPM(parameter_set=parameter_set)

# Construct the fitting parameters
# with initial values sampled from a different distribution
parameters = pybop.Parameters(
    pybop.Parameter(
        "Negative electrode active material volume fraction",
        prior=pybop.Uniform(0.3, 0.9),
        bounds=[0.3, 0.8],
        initial_value=pybop.Uniform(0.4, 0.75).rvs()[0],
        true_value=parameter_set["Negative electrode active material volume fraction"],
    ),
    pybop.Parameter(
        "Positive electrode active material volume fraction",
        prior=pybop.Uniform(0.3, 0.9),
        initial_value=pybop.Uniform(0.4, 0.75).rvs()[0],
        true_value=parameter_set["Positive electrode active material volume fraction"],
        # no bounds
    ),
)

# Generate data
sigma = 0.002
experiment = pybop.Experiment(
    [
        (
            "Discharge at 0.5C for 12 minutes (10 second period)",
            "Charge at 0.5C for 12 minutes (10 second period)",
        )
    ]
)
values = model.predict(initial_state={"Initial SoC": 0.4}, experiment=experiment)


def noise(sigma):
    return np.random.normal(0, sigma, len(values["Voltage [V]"].data))


# Form dataset
dataset = pybop.Dataset(
    {
        "Time [s]": values["Time [s]"].data,
        "Current function [A]": values["Current [A]"].data,
        "Voltage [V]": values["Voltage [V]"].data + noise(sigma),
    }
)

# Generate problem, cost function, and optimisation class
problem = pybop.FittingProblem(model, parameters, dataset)
likelihood = pybop.GaussianLogLikelihoodKnownSigma(problem, sigma0=sigma * 1.05)
posterior = pybop.LogPosterior(likelihood)
optim = pybop.IRPropMin(
    posterior, max_iterations=125, max_unchanged_iterations=60, sigma0=0.01
)

results = optim.run()
print(parameters.true_value())

# Plot the timeseries output
pybop.plot.quick(problem, problem_inputs=results.x, title="Optimised Comparison")

# Plot convergence
pybop.plot.convergence(optim)

# Plot the parameter traces
pybop.plot.parameters(optim)

# Contour plot with optimisation path
bounds = np.asarray([[0.5, 0.8], [0.4, 0.7]])
pybop.plot.surface(optim, bounds=bounds)
