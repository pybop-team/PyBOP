import numpy as np

import pybop

# Import the ECM parameter set from JSON
parameter_set = pybop.ParameterSet(
    json_path="examples/scripts/parameters/initial_ecm_parameters.json"
)
parameter_set.import_parameters()

# Alternatively, define the initial parameter set with a dictionary
# Add definitions for R's, C's, and initial overpotentials for any additional RC elements
# parameter_set = pybop.ParameterSet(
#     params_dict={
#         "chemistry": "ecm",
#         "Initial SoC": 0.5,
#         "Initial temperature [K]": 25 + 273.15,
#         "Cell capacity [A.h]": 5,
#         "Nominal cell capacity [A.h]": 5,
#         "Ambient temperature [K]": 25 + 273.15,
#         "Current function [A]": 5,
#         "Upper voltage cut-off [V]": 4.2,
#         "Lower voltage cut-off [V]": 3.0,
#         "Cell thermal mass [J/K]": 1000,
#         "Cell-jig heat transfer coefficient [W/K]": 10,
#         "Jig thermal mass [J/K]": 500,
#         "Jig-air heat transfer coefficient [W/K]": 10,
#         "Open-circuit voltage [V]": pybop.empirical.Thevenin().default_parameter_values[
#             "Open-circuit voltage [V]"
#         ],
#         "R0 [Ohm]": 0.001,
#         "Element-1 initial overpotential [V]": 0,
#         "Element-2 initial overpotential [V]": 0,
#         "R1 [Ohm]": 0.0002,
#         "R2 [Ohm]": 0.0003,
#         "C1 [F]": 10000,
#         "C2 [F]": 5000,
#         "Entropic change [V/K]": 0.0004,
#     }
# )

# Define the model
model = pybop.empirical.Thevenin(
    parameter_set=parameter_set, options={"number of rc elements": 2}
)

# Fitting parameters
parameters = pybop.Parameters(
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
)

sigma = 0.001
t_eval = np.arange(0, 900, 3)
values = model.predict(t_eval=t_eval)
corrupt_values = values["Voltage [V]"].data + np.random.normal(0, sigma, len(t_eval))

# Form dataset
dataset = pybop.Dataset(
    {
        "Time [s]": t_eval,
        "Current function [A]": values["Current [A]"].data,
        "Voltage [V]": corrupt_values,
    }
)

# Generate problem, cost function, and optimisation class
problem = pybop.FittingProblem(model, parameters, dataset)
cost = pybop.SumSquaredError(problem)
optim = pybop.CMAES(cost, max_iterations=100)

x, final_cost = optim.run()
print("Estimated parameters:", x)

# Export the parameters to JSON
parameter_set.export_parameters(
    "examples/scripts/parameters/fit_ecm_parameters.json", fit_params=parameters
)

# Plot the time series
pybop.plot_dataset(dataset)

# Plot the timeseries output
pybop.quick_plot(problem, parameter_values=x, title="Optimised Comparison")

# Plot convergence
pybop.plot_convergence(optim)

# Plot the parameter traces
pybop.plot_parameters(optim)

# Plot the cost landscape
pybop.plot2d(cost, steps=15)

# Plot the cost landscape with optimisation path and updated bounds
bounds = np.array([[1e-4, 1e-2], [1e-5, 1e-2]])
pybop.plot2d(optim, bounds=bounds, steps=15)
