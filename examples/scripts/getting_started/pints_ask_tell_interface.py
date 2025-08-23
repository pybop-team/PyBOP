import matplotlib.pyplot as plt
import numpy as np
import pybamm

import pybop

"""
In this example, we will introduce the ask-tell optimiser interface for the PINTS-based
optimisers. This interface provides a simple method for flexible optimisation workflows.
"""

# Define model and parameter values
model = pybamm.lithium_ion.SPM()
parameter_values = pybamm.ParameterValues("Chen2020")

# Generate a synthetic dataset
sim = pybamm.Simulation(model, parameter_values=parameter_values)
sol = sim.solve(t_eval=np.linspace(0, 100, 100))
dataset = pybop.Dataset(
    {
        "Time [s]": sol.t,
        "Current function [A]": sol["Current [A]"].data,
        "Voltage [V]": sol["Voltage [V]"].data,
        "Bulk open-circuit voltage [V]": sol["Bulk open-circuit voltage [V]"].data,
    }
)

# Fitting parameters
parameters = [
    pybop.Parameter(
        "Negative electrode active material volume fraction",
        prior=pybop.Gaussian(0.55, 0.05),
    ),
    pybop.Parameter(
        "Positive electrode active material volume fraction",
        prior=pybop.Gaussian(0.55, 0.05),
    ),
]

# Construct the problem class
builder = (
    pybop.builders.Pybamm()
    .set_dataset(dataset)
    .set_simulation(model, parameter_values=parameter_values)
    .add_cost(pybop.costs.pybamm.Minkowski("Voltage [V]", "Voltage [V]"))
    .add_cost(
        pybop.costs.pybamm.Minkowski(
            "Bulk open-circuit voltage [V]", "Bulk open-circuit voltage [V]", p=2
        ),
        weight=0.5,
    )
)
for param in parameters:
    builder.add_parameter(param)
problem = builder.build()

# We construct the optimiser class the same as normal but will be using the `optimiser`
# attribute directly for this example. This interface works for all PINTS-based
# optimisers. Warning: not all arguments are supported via this interface.
options = pybop.PintsOptions(sigma=1e-2)
optim = pybop.AdamW(problem, options=options)

# Create storage vars
x_best = []
f_best = []

# Run optimisation
for i in range(70):
    x = optim.optimiser.ask()
    problem.set_params(x)
    f = [problem.run_with_sensitivities()]
    optim.optimiser.tell(f)

    # Store best solution so far
    x_best.append(optim.optimiser.x_best())
    f_best.append(optim.optimiser.x_best())

    if i % 10 == 0:
        print(
            f"Iteration: {i} | Cost: {optim.optimiser.f_best()} | Parameters: {optim.optimiser.x_best()}"
        )

# Manually apply the optimal parameters to the ParameterValues object
# Next, we solve the forward model with the PyBaMM Simulation class
for i, param in enumerate(problem.params):
    parameter_values.update({param.name: x_best[-1][i]})
sim = pybamm.Simulation(model, parameter_values=parameter_values)
sol = sim.solve(t_eval=[dataset["Time [s]"][0], dataset["Time [s]"][-1]])

# Plot the timeseries output
fig, ax = plt.subplots()
ax.plot(dataset["Time [s]"], dataset["Voltage [V]"], label="Targe")
ax.plot(sol.t, sol["Voltage [V]"].data, label="Fit")
ax.set(xlabel="Time (s)", ylabel="Voltage [V]")
ax.legend()
ax.grid()
plt.show()
