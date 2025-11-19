import os

import numpy as np
import pybamm

import pybop

# Get the current directory location and convert to absolute path
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(
    current_dir, "../../data/synthetic/dfn_charge_discharge_75.csv"
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
model = pybamm.lithium_ion.DFN()
parameter_values = pybamm.ParameterValues("Chen2020")
parameter_values.set_initial_state(f"{csv_data[0, 2]} V")

# Fitting parameters
parameter_values.update(
    {
        "Negative electrode active material volume fraction": pybop.ParameterDistribution(
            pybop.Gaussian(
                0.68,
                0.05,
                truncated_at=[0.4, 0.9],
            ),
            initial_value=0.65,
        ),
        "Positive electrode active material volume fraction": pybop.ParameterDistribution(
            pybop.Gaussian(
                0.58,
                0.05,
                truncated_at=[0.4, 0.9],
            ),
            initial_value=0.65,
        ),
    }
)

# Build the problem
simulator = pybop.pybamm.Simulator(model, parameter_values, protocol=dataset)
target = ["Voltage [V]", "Bulk open-circuit voltage [V]"]
cost = pybop.RootMeanSquaredError(dataset, target=target)
problem = pybop.Problem(simulator, cost)

# Set up the optimiser
options = pybop.PintsOptions(
    max_iterations=60,
    max_unchanged_iterations=15,
)
optim = pybop.IRPropPlus(problem, options=options)

# Run the optimisation
result = optim.run()
print(result)

# Plot the timeseries output
pybop.plot.problem(problem, inputs=result.best_inputs, title="Optimised Comparison")

# Plot the optimisation result
result.plot_convergence()
result.plot_parameters()
result.plot_surface()
