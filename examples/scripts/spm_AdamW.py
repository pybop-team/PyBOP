import numpy as np

import pybop

# Parameter set and model definition
parameter_set = pybop.ParameterSet.pybamm("Chen2020")
model = pybop.lithium_ion.SPM(parameter_set=parameter_set)

# Fitting parameters
parameters = pybop.Parameters(
    pybop.Parameter(
        "Negative electrode active material volume fraction",
        prior=pybop.Gaussian(0.68, 0.05),
    ),
    pybop.Parameter(
        "Positive electrode active material volume fraction",
        prior=pybop.Gaussian(0.58, 0.05),
    ),
)

# Generate data
init_soc = 0.5
sigma = 0.003
experiment = pybop.Experiment(
    [
        (
            "Discharge at 0.5C for 3 minutes (3 second period)",
            "Charge at 0.5C for 3 minutes (3 second period)",
        ),
    ]
    * 2
)
values = model.predict(init_soc=init_soc, experiment=experiment)


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

signal = ["Voltage [V]", "Bulk open-circuit voltage [V]"]
# Generate problem, cost function, and optimisation class
problem = pybop.FittingProblem(
    model, parameters, dataset, signal=signal, init_soc=init_soc
)
cost = pybop.Minkowski(problem, p=2)
optim = pybop.AdamW(
    cost,
    verbose=True,
    allow_infeasible_solutions=True,
    sigma0=0.05,
    max_iterations=100,
    max_unchanged_iterations=20,
)

# Run optimisation
x, final_cost = optim.run()
print("Estimated parameters:", x)

# Plot the timeseries output
pybop.quick_plot(problem, problem_inputs=x, title="Optimised Comparison")

# Plot convergence
pybop.plot_convergence(optim)

# Plot the parameter traces
pybop.plot_parameters(optim)

# Plot the cost landscape with optimisation path
bounds = np.asarray([[0.5, 0.8], [0.4, 0.7]])
pybop.plot2d(optim, bounds=bounds, steps=15)
