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
        prior=pybop.Gaussian(0.55, 0.05),
    ),
    pybop.Parameter(
        "Positive electrode active material volume fraction",
        prior=pybop.Gaussian(0.55, 0.05),
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

signal = ["Voltage [V]", "Bulk open-circuit voltage [V]"]
# Construct the problem and cost classes
problem = pybop.FittingProblem(model, parameters, dataset, signal=signal)
cost = pybop.Minkowski(problem, p=2)

# We construct the optimiser class the same as normal
# but will be using the `optimiser` attribute directly
# for this example. This interface works for all
# non SciPy-based optimisers.
# Warning: not all arguments are supported via this
# interface.
optim = pybop.AdamW(cost)

# Create storage vars
x_best = []
f_best = []

# Run optimisation
for i in range(100):
    x = optim.optimiser.ask()
    f = [cost(x[0], calculate_grad=True)]
    optim.optimiser.tell(f)

    # Store best solution so far
    x_best.append(optim.optimiser.x_best())
    f_best.append(optim.optimiser.x_best())

    if i % 10 == 0:
        print(
            f"Iteration: {i} | Cost: {optim.optimiser.f_best()} | Parameters: {optim.optimiser.x_best()}"
        )

# Plot the timeseries output
pybop.plot.problem(
    problem, problem_inputs=optim.optimiser.x_best(), title="Optimised Comparison"
)
