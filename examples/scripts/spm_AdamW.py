import numpy as np
import pybamm

import pybop

# Define model and use high-performant solver for sensitivities
solver = pybamm.IDAKLUSolver()
parameter_set = pybop.ParameterSet.pybamm("Chen2020")
model = pybop.lithium_ion.SPM(parameter_set=parameter_set, solver=solver)

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
values = model.predict(initial_state={"Initial SoC": 0.5}, experiment=experiment)


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
problem = pybop.FittingProblem(model, parameters, dataset, signal=signal)
cost = pybop.Minkowski(problem, p=2)
optim = pybop.AdamW(
    cost,
    verbose=True,
    allow_infeasible_solutions=True,
    sigma0=0.02,
    max_iterations=100,
    max_unchanged_iterations=20,
)

# Run optimisation
results = optim.run()
print(results)

# Plot the timeseries output
pybop.plot.quick(problem, problem_inputs=results.x, title="Optimised Comparison")

# Plot convergence
pybop.plot.convergence(optim)

# Plot the parameter traces
pybop.plot.parameters(optim)

# Plot the cost landscape with optimisation path
pybop.plot.surface(optim)
