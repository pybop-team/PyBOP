import numpy as np

import pybop

# Define model
parameter_set = pybop.ParameterSet.pybamm("Chen2020")
model = pybop.lithium_ion.SPM(parameter_set=parameter_set)

# Fitting parameters
parameters = pybop.Parameters(
    pybop.Parameter(
        "Negative electrode active material volume fraction",
        prior=pybop.Gaussian(0.6, 0.05),
        bounds=[0.4, 0.75],
        initial_value=0.41,
        true_value=0.7,
    ),
    pybop.Parameter(
        "Positive electrode active material volume fraction",
        prior=pybop.Gaussian(0.48, 0.05),
        bounds=[0.4, 0.75],
        initial_value=0.41,
        true_value=0.67,
    ),
)
init_soc = 0.7
experiment = pybop.Experiment(
    [
        (
            "Discharge at 0.5C for 3 minutes (4 second period)",
            "Charge at 0.5C for 3 minutes (4 second period)",
        ),
    ]
)
values = model.predict(
    init_soc=init_soc, experiment=experiment, inputs=parameters.as_dict("true")
)

sigma = 0.002
corrupt_values = values["Voltage [V]"].data + np.random.normal(
    0, sigma, len(values["Voltage [V]"].data)
)

# Form dataset
dataset = pybop.Dataset(
    {
        "Time [s]": values["Time [s]"].data,
        "Current function [A]": values["Current [A]"].data,
        "Voltage [V]": corrupt_values,
    }
)

# Generate problem, cost function, and optimisation class
problem = pybop.FittingProblem(model, parameters, dataset, init_soc=init_soc)
cost = pybop.GaussianLogLikelihood(problem, sigma0=sigma * 4)
optim = pybop.Optimisation(
    cost,
    optimiser=pybop.CuckooSearch,
    max_iterations=100,
)

x, final_cost = optim.run()
print("Estimated parameters:", x)

# Plot the timeseries output
pybop.quick_plot(problem, problem_inputs=x, title="Optimised Comparison")

# Plot convergence
pybop.plot_convergence(optim)

# Plot the parameter traces
pybop.plot_parameters(optim)

# Plot the cost landscape with optimisation path
pybop.plot2d(optim, steps=15)
