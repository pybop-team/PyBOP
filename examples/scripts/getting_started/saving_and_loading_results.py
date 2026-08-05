from pathlib import Path

import numpy as np
import pybamm

import pybop

"""
This example shows how to save and load an optimisation (or sampling) result.
First we run an example optimisation to generate a result.
"""

# Define model and parameter values
model = pybamm.lithium_ion.SPM()
parameter_values = pybamm.ParameterValues("Chen2020")

# Generate a synthetic dataset
sigma = 5e-3
t_eval = np.linspace(0, 500, 240)
solution = pybamm.Simulation(model, parameter_values=parameter_values).solve(
    t_eval=t_eval
)
dataset = pybop.Dataset(
    {
        "Time [s]": t_eval,
        "Current [A]": solution["Current [A]"](t_eval),
        "Voltage [V]": pybop.add_noise(solution["Voltage [V]"](t_eval), sigma),
    }
)

# Fitting parameters
parameter_values.update(
    {
        "Negative electrode active material volume fraction": pybop.Parameter(
            distribution=pybop.Gaussian(0.68, 0.05, truncated_at=[0.4, 0.9]),
            initial_value=0.45,
        ),
        "Positive electrode active material volume fraction": pybop.Parameter(
            distribution=pybop.Gaussian(0.58, 0.05, truncated_at=[0.4, 0.9]),
            initial_value=0.45,
        ),
    }
)

# Build the problem
simulator = pybop.pybamm.Simulator(
    model, parameter_values=parameter_values, protocol=dataset
)
cost = pybop.MeanAbsoluteError(dataset)
problem = pybop.Problem(simulator, cost)

# Set up the optimiser
options = pybop.PintsOptions(max_iterations=150, verbose=True)
optim = pybop.PSO(problem, options=options)

# Run the optimisation
result = optim.run()

# Define paths relative to the script's directory
save_path = Path(__file__).parent / "results"
save_path.mkdir(exist_ok=True)

# Save the result: either pickle the whole result or save the data in
# one of these formats: "pickle", "json", "matlab"
result.save(save_path / "saved_result_object.pkl")
result.save_data(save_path / "saved_result_data.json", to_format="json")

# Load the result
result_from_pkl = pybop.Result.load(save_path / "saved_result_object.pkl")
result_from_json = pybop.Result.load_data(
    save_path / "saved_result_data.json", file_format="json"
)

# Plot the optimisation result from .pkl
result_from_pkl.plot_convergence()
result_from_pkl.plot_parameters()
result_from_pkl.plot_surface(bounds=problem.parameters.get_bounds_array())

# Plot the optimisation result from .json (it is the same)
result_from_json.plot_convergence()
result_from_json.plot_parameters()
result_from_json.plot_surface(bounds=problem.parameters.get_bounds_array())
