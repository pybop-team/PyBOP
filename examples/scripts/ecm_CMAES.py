import pybop
import numpy as np

# Import the ECM parameter set from JSON
params = pybop.ParameterSet(
    json_path="examples/scripts/parameters/initial_ecm_parameters.json"
)

# Define the model
model = pybop.empirical.Thevenin(
    parameter_set=params.import_parameters(), options={"number of rc elements": 2}
)

# Fitting parameters
parameters = [
    pybop.Parameter(
        "R0 [Ohm]",
        prior=pybop.Gaussian(0.0002, 0.0001),
        bounds=[1e-4, 1e-2],
    ),
    pybop.Parameter(
        "R1 [Ohm]",
        prior=pybop.Gaussian(0.0001, 0.0001),
        bounds=[1e-5, 1e-2],
    ),
]

sigma = 0.001
t_eval = np.arange(0, 900, 2)
values = model.predict(t_eval=t_eval)
CorruptValues = values["Voltage [V]"].data + np.random.normal(0, sigma, len(t_eval))

dataset = [
    pybop.Dataset("Time [s]", t_eval),
    pybop.Dataset("Current function [A]", values["Current [A]"].data),
    pybop.Dataset("Voltage [V]", CorruptValues),
]

# Generate problem, cost function, and optimisation class
problem = pybop.FittingProblem(model, parameters, dataset)
cost = pybop.SumSquaredError(problem)
optim = pybop.Optimisation(cost, optimiser=pybop.CMAES)
optim.set_max_iterations(100)

x, final_cost = optim.run()
print("Estimated parameters:", x)

# Export the parameters to JSON
params.export_parameters(
    "examples/scripts/parameters/fit_ecm_parameters.json", fit_params=parameters
)

# Plot the timeseries output
pybop.quick_plot(x, cost, title="Optimised Comparison")

# Plot convergence
pybop.plot_convergence(optim)

# Plot the parameter traces
pybop.plot_parameters(optim)

# Plot the cost landscape
pybop.plot_cost2d(cost, steps=15)

# Plot the cost landscape with optimisation path and updated bounds
bounds = np.array([[1e-4, 1e-2], [1e-5, 1e-2]])
pybop.plot_cost2d(cost, optim=optim, bounds=bounds, steps=15)
