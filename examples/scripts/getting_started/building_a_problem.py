import numpy as np
import pybamm

import pybop

model = pybamm.lithium_ion.SPM()
parameter_values = pybamm.ParameterValues("Chen2020")
t_eval = np.linspace(0, 100, 240)
sim = pybamm.Simulation(model=model, parameter_values=parameter_values)
sol = sim.solve(t_eval=t_eval)

dataset = pybop.Dataset(
    {
        "Time [s]": sol.t,
        "Voltage [V]": sol["Voltage [V]"].data,
        "Current function [A]": sol["Current [A]"].data,
    }
)

# Create the builder
builder = pybop.builders.Pybamm()
builder.set_dataset(dataset)
builder.set_simulation(
    model,
    parameter_values=parameter_values,
)
builder.add_parameter(
    {
        "name": "Negative electrode active material volume fraction",
        "initial_value": 0.6,
        "bounds": [0.5, 0.8],
    }
)
builder.add_parameter(
    {
        "name": "Positive electrode active material volume fraction",
        "initial_value": 0.6,
        "bounds": [0.5, 0.8],
    }
)
builder.add_cost(
    pybop.costs.pybamm.NegativeGaussianLogLikelihood("Voltage [V]", "Voltage [V]")
)

# Build the problem
problem = builder.build()

options = pybop.PintsOptions(max_iterations=3, parallel=True)
optim = pybop.CMAES(problem, options=options)
optim.set_population_size(1000)
results = optim.run()
print(results)

# # Solve
# problem.set_params(np.array([0.6, 0.6]))
# sol = problem.pipeline.solve()
#
# # Plot
# fig, ax = plt.subplots()
# ax.scatter(sol["Time [s]"].data, sol["Voltage [V]"].data)
# plt.show()
