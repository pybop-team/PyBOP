import numpy as np
import pybamm

import pybop

# Parameter set definition
parameter_set = pybop.ParameterSet("Chen2020")
parameter_set["Lower voltage cut-off [V]"] = 2.3
parameter_set["Upper voltage cut-off [V]"] = 4.4

# Set initial state and unpack true values
parameter_set.parameter_values.set_initial_stoichiometries(initial_value=1.0)
cs_n_max = parameter_set["Maximum concentration in negative electrode [mol.m-3]"]
cs_p_max = parameter_set["Maximum concentration in positive electrode [mol.m-3]"]
cs_n_init = parameter_set["Initial concentration in negative electrode [mol.m-3]"]
cs_p_init = parameter_set["Initial concentration in positive electrode [mol.m-3]"]

# Model definition
model = pybop.lithium_ion.SPM(parameter_set=parameter_set)

# Define fitting parameters for OCP balancing
parameters = pybop.Parameters(
    pybop.Parameter(
        "Maximum concentration in negative electrode [mol.m-3]",
        prior=pybop.Gaussian(cs_n_max, 6e3),
        bounds=[cs_n_max * 0.75, cs_n_max * 1.25],
        true_value=cs_n_max,
        initial_value=cs_n_max * 0.8,
    ),
    pybop.Parameter(
        "Maximum concentration in positive electrode [mol.m-3]",
        prior=pybop.Gaussian(cs_p_max, 6e3),
        bounds=[cs_p_max * 0.75, cs_p_max * 1.25],
        true_value=cs_p_max,
        initial_value=cs_p_max * 0.8,
    ),
    pybop.Parameter(
        "Initial concentration in negative electrode [mol.m-3]",
        prior=pybop.Gaussian(cs_n_init, 6e3),
        bounds=[cs_n_max * 0.75, cs_n_max * 1.25],
        true_value=cs_n_init,
        initial_value=cs_n_max * 0.8,
    ),
    pybop.Parameter(
        "Initial concentration in positive electrode [mol.m-3]",
        prior=pybop.Gaussian(cs_p_init, 6e3),
        bounds=[0, cs_p_max * 0.5],
        true_value=cs_p_init,
        initial_value=cs_p_max * 0.2,
    ),
)

# Generate synthetic data
sigma = 5e-4  # Volts
experiment = pybamm.Experiment(
    [
        "Discharge at 0.1C until 2.5V (3 min period)",
        "Charge at 0.1C until 4.2V (3 min period)",
    ]
)
values = model.predict(experiment=experiment)


def noisy(data, sigma):
    return data + np.random.normal(0, sigma, len(data))


# Form dataset
dataset = pybop.Dataset(
    {
        "Time [s]": values["Time [s]"].data,
        "Current function [A]": values["Current [A]"].data,
        "Voltage [V]": noisy(values["Voltage [V]"].data, sigma),
    }
)

# Generate problem, cost function, and optimisation class
problem = pybop.FittingProblem(model, parameters, dataset)
cost = pybop.GaussianLogLikelihoodKnownSigma(problem, sigma0=sigma)
optim = pybop.SciPyMinimize(cost, max_iterations=125)

# Run optimisation for Maximum Likelihood Estimate (MLE)
results = optim.run()
print("True parameters:", parameters.true_value())

# Plot the timeseries output
pybop.plot.problem(problem, problem_inputs=results.x, title="Optimised Comparison")

# Plot convergence
pybop.plot.convergence(optim)

# Plot the parameter traces
pybop.plot.parameters(optim)

# Plot the cost landscape with optimisation path
pybop.plot.contour(optim, steps=5)
