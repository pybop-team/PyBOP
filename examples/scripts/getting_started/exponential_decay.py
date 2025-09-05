import numpy as np
import pybamm

import pybop

# Define model and parameter values
model = pybop.ExponentialDecayModel()
parameter_values = pybamm.ParameterValues({"k": 0.3, "y0": 1.5})

# Fitting parameters
parameters = [
    pybop.Parameter("k", prior=pybop.Gaussian(0.5, 0.05), bounds=[0, 2]),
    pybop.Parameter("y0", prior=pybop.Gaussian(1.0, 0.05), bounds=[0, 2]),
]

# Generate a synthetic dataset
sim = pybamm.Simulation(model, parameter_values=parameter_values)
t_eval = np.linspace(0, 10, 50)
sol = sim.solve(t_eval=t_eval)

dataset = pybop.Dataset(
    {
        "Time [s]": t_eval,
        "y_0": sol["y_0"](t_eval),
    }
)

# Construct the problem builder
builder = (
    pybop.builders.Pybamm()
    .set_dataset(dataset)
    .set_simulation(model, parameter_values=parameter_values)
    .add_cost(pybop.costs.pybamm.MeanSquaredError("y_0"))
)
for param in parameters:
    builder.add_parameter(param)
problem = builder.build()

# Set optimiser and options
options = pybop.PintsOptions(sigma=0.2, verbose=True, max_iterations=100)
optim = pybop.CMAES(problem, options=options)
# optim.set_population_size(200)
# optim = pybop.SciPyMinimize(problem)
# optim = pybop.IRPropPlus(problem)

# Run optimisation
results = optim.run()

# Plot convergence
results.plot_convergence()

# Plot the parameter traces
results.plot_parameters()

# Plot the cost landscape with optimisation path
results.plot_surface()

# Compare the fit to the data
pybop.plot.validation(results.x, problem=problem, signal="y_0", dataset=dataset)
