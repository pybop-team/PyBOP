import os

import numpy as np
import pybamm

import pybop

# Get the current directory location and convert to absolute path
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(
    current_dir, "../../data/synthetic/spm_charge_discharge_75.csv"
)

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

# Import the synthetic dataset
csv_data = np.loadtxt(dataset_path, delimiter=",", skiprows=1)

# Form dataset
dataset = pybop.Dataset(
    {
        "Time [s]": csv_data[:, 0],
        "Current function [A]": csv_data[:, 1],
        "Voltage [V]": csv_data[:, 2],
        "Bulk open-circuit voltage [V]": csv_data[:, 3],
    }
)


signal = ["Voltage [V]", "Bulk open-circuit voltage [V]"]
# Construct the problem and cost classes
problem = pybop.FittingProblem(
    model,
    parameters,
    dataset,
    signal=signal,
    initial_state={"Initial open-circuit voltage [V]": csv_data[0, 2]},
)
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
pybop.plot.quick(
    problem, problem_inputs=optim.optimiser.x_best(), title="Optimised Comparison"
)
