import os

import numpy as np
import pybamm

import pybop

# Get the current directory location and convert to absolute path
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(
    current_dir, "../../data/synthetic/spm_charge_discharge_75.csv"
)

# Import the synthetic dataset
csv_data = np.loadtxt(dataset_path, delimiter=",", skiprows=1)
dataset = pybop.Dataset(
    {
        "Time [s]": csv_data[:, 0],
        "Current function [A]": csv_data[:, 1],
        "Voltage [V]": csv_data[:, 2],
        "Bulk open-circuit voltage [V]": csv_data[:, 3],
    }
)

# Define model and parameter values
model = pybamm.lithium_ion.SPM()
parameter_values = pybamm.ParameterValues("Chen2020")
parameter_values.set_initial_state(f"{csv_data[0, 2]} V")

# Fitting parameters
parameter_values.update(
    {
        "Negative electrode active material volume fraction": pybop.Gaussian(
            0.55, 0.05
        ),
        "Positive electrode active material volume fraction": pybop.Gaussian(
            0.55, 0.05
        ),
    }
)

# Build the problem
simulator = pybop.pybamm.Simulator(
    model, parameter_values=parameter_values, protocol=dataset
)
cost = pybop.Minkowski(
    dataset, target=["Voltage [V]", "Bulk open-circuit voltage [V]"], p=2
)
problem = pybop.Problem(simulator, cost)

# We construct the optimiser class the same as normal but will be using the
# `optimiser` attribute directly for this example. This interface works for
# all PINTS-based optimisers.
# Warning: not all arguments are supported via this interface.
options = pybop.PintsOptions(verbose=True)
optim = pybop.AdamW(problem, options=options)

# Create storage vars
x_best = []
f_best = []

# Run the optimisation
for i in range(50):
    x = optim.optimiser.ask()
    f = [problem.evaluate(x[0], calculate_sensitivities=True).get_values()]
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
    problem, problem_inputs=optim.optimiser.x_best()[0], title="Optimised Comparison"
)
