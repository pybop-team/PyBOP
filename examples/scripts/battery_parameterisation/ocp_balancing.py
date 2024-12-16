import numpy as np

import pybop

# Parameter set and model definition
parameter_set = pybop.ParameterSet.pybamm("Chen2020")
# solver = pybamm.IDAKLUSolver()
model = pybop.lithium_ion.SPM(parameter_set=parameter_set)  # , solver=solver)

# Set objects for initial conditions
init_soc = {"Initial SoC": 1}
parameter_set.set_initial_stoichiometries(initial_value=init_soc["Initial SoC"])
parameter_set["Lower voltage cut-off [V]"] = -1e9
parameter_set["Upper voltage cut-off [V]"] = 1e9
cs_n_max = parameter_set["Maximum concentration in negative electrode [mol.m-3]"]
cs_p_max = parameter_set["Maximum concentration in positive electrode [mol.m-3]"]
v_min = parameter_set["Open-circuit voltage at 0% SOC [V]"]
v_max = parameter_set["Open-circuit voltage at 100% SOC [V]"]

# Define fitting parameters for OCP balancing
parameters = pybop.Parameters(
    pybop.Parameter(
        "Open-circuit voltage at 0% SOC [V]",
        prior=pybop.Gaussian(v_min, 1e-1),
        bounds=[2.35, 2.65],
        true_value=v_min,
        transformation=pybop.LogTransformation(),
    ),
    pybop.Parameter(
        "Open-circuit voltage at 100% SOC [V]",
        prior=pybop.Gaussian(v_max, 1e-1),
        bounds=[4.1, 4.3],
        true_value=v_max,
        transformation=pybop.LogTransformation(),
    ),
    pybop.Parameter(
        "Maximum concentration in negative electrode [mol.m-3]",
        prior=pybop.Gaussian(cs_n_max, 6e3),
        bounds=[cs_n_max * 0.75, cs_n_max * 1.25],
        true_value=cs_n_max,
        transformation=pybop.LogTransformation(),
    ),
    pybop.Parameter(
        "Maximum concentration in positive electrode [mol.m-3]",
        prior=pybop.Gaussian(cs_p_max, 6e3),
        bounds=[cs_p_max * 0.75, cs_p_max * 1.25],
        true_value=cs_p_max,
        transformation=pybop.LogTransformation(),
    ),
)

# Generate synthetic data
sigma = 5e-4  # Volts
experiment = pybop.Experiment(
    [
        (
            "Discharge at 0.1C until 2.5V (3 min period)",
            "Charge at 0.1C until 4.2V (3 min period)",
        ),
    ]
)
values = model.predict(initial_state=init_soc, experiment=experiment)


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
problem = pybop.FittingProblem(model, parameters, dataset, initial_state=init_soc)
cost = pybop.GaussianLogLikelihoodKnownSigma(problem, sigma0=sigma)
optim = pybop.CMAES(
    cost,
    verbose=True,
    sigma0=1e-1,
    max_iterations=100,
    max_unchanged_iterations=20,
)

# Run optimisation for Maximum Likelihood Estimate (MLE)
results = optim.run()
print("True parameters:", parameters.true_value())

# Plot the timeseries output
pybop.plot.quick(problem, problem_inputs=results.x, title="Optimised Comparison")

# Plot convergence
pybop.plot.convergence(optim)

# Plot the parameter traces
pybop.plot.parameters(optim)

# Plot the cost landscape with optimisation path
pybop.plot.contour(optim, steps=5)
