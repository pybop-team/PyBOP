import numpy as np
import pybamm

import pybop

"""
In this example, we demonstrate identification using a Doyle-Fuller-Newman (DFN) model.
The DFN is more challenging to parameterise than equivalent circuit and reduced-order,
single particle models as it is more computationally expensive and has a high number of
parameters which are not individually identifiable.
"""


# Define model and parameter values
model = pybamm.lithium_ion.DFN()
parameter_values = pybamm.ParameterValues("Chen2020")

# Generate a synthetic dataset
sim = pybamm.Simulation(model, parameter_values=parameter_values)
t_eval = np.linspace(0, 500, 240)
solution = sim.solve(t_eval=t_eval)
sigma = 5e-3
dataset = pybop.Dataset(
    {
        "Time [s]": t_eval,
        "Voltage [V]": solution["Voltage [V]"](t_eval)
        + np.random.normal(0, sigma, len(t_eval)),
        "Current function [A]": solution["Current [A]"](t_eval),
        "Bulk open-circuit voltage [V]": solution["Bulk open-circuit voltage [V]"](
            t_eval
        ),
    }
)

# Save the true values
true_values = [
    parameter_values[p]
    for p in [
        "Negative electrode active material volume fraction",
        "Positive electrode active material volume fraction",
    ]
]

# Fitting parameters
parameter_values.update(
    {
        "Negative electrode active material volume fraction": pybop.ParameterInfo(
            pybop.Gaussian(
                0.68,
                0.05,
                truncated_at=[0.4, 0.9],
            ),
            initial_value=0.65,
        ),
        "Positive electrode active material volume fraction": pybop.ParameterInfo(
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

# Compare identified to true parameter values
print("True parameters:", true_values)
print("Identified parameters:", result.x)

# Plot the timeseries output
pybop.plot.problem(problem, inputs=result.best_inputs, title="Optimised Comparison")

# Plot the optimisation result
result.plot_convergence()
result.plot_parameters()
result.plot_surface()
