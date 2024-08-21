
# A script to generate parameterisation plots for the JOSS paper.

import pybop
import numpy as np


# Parameter set and model definition
parameter_set = pybop.ParameterSet.pybamm("Chen2020")
options={"surface form": "differential", "contact resistance": "true"}
parameter_set["Contact resistance [Ohm]"] = 0.01
model = pybop.lithium_ion.SPM(parameter_set=parameter_set, options=options)

# Generate input dataset
experiment = pybop.Experiment([("Discharge at 1C until 2.5 V", "Rest for 30 minutes")])

# Generate data and add Gaussian noise
values = model.predict(experiment=experiment)
sigma = 0.01
corrupt_values = values["Voltage [V]"].data + np.random.normal(0, sigma, len(values["Voltage [V]"].data))

# Form dataset
dataset = pybop.Dataset(
    {
        "Time [s]": values["Time [s]"].data,
        "Current function [A]": values["Current [A]"].data,
        "Voltage [V]": corrupt_values,
    }
)

# Fitting parameters
parameters = pybop.Parameters(
    pybop.Parameter(
        "Negative particle diffusivity [m2.s-1]",
        initial_value=5e-14,
        prior=pybop.Gaussian(5e-14, 0.5e-14),
        # transformation=pybop.LogTransformation(),
        bounds=[1.9e-14, 12e-14],
        true_value=parameter_set["Negative particle diffusivity [m2.s-1]"],
    ),
    pybop.Parameter(
        "Contact resistance [Ohm]",
        initial_value=0.015,
        prior=pybop.Gaussian(0.015, 0.005),
        bounds=[0.0049, 0.025],
        true_value=parameter_set["Contact resistance [Ohm]"],
    ),
)

# Generate problem, cost and optimiser classes
problem = pybop.FittingProblem(model, parameters, dataset)
cost = pybop.SumSquaredError(problem)
optim = pybop.NelderMead(
    cost,
    verbose=True,
    max_iterations=500,
    max_unchanged_iterations=50,
)

# Run optimisation
x, final_cost = optim.run()
print("True parameter values:", parameters.true_value())
print("Estimated parameters:", x)

# Plot the timeseries output
pybop.quick_plot(problem, problem_inputs=x, title="Optimised Comparison")

# Plot convergence
pybop.plot_convergence(optim)

# Plot the parameter traces
pybop.plot_parameters(optim)

# Plot the cost landscape with optimisation path
pybop.plot2d(optim, steps=15)
