import numpy as np

import pybop
from pybop.models.lithium_ion.basic_SP_diffusion import (
    convert_physical_to_electrode_parameters,
)

# Define model
parameter_set = pybop.ParameterSet("Xu2019")
model = pybop.lithium_ion.SPMe(
    parameter_set=parameter_set, options={"working electrode": "positive"}
)

# Generate data
sigma = 1e-3
initial_state = {"Initial SoC": 0.9}
experiment = pybop.Experiment(
    [
        "Rest for 1 second",
        "Discharge at 1C for 10 minutes (10 second period)",
        "Rest for 20 minutes",
    ]
)
values = model.predict(initial_state=initial_state, experiment=experiment)
corrupt_values = values["Voltage [V]"].data + np.random.normal(
    0, sigma, len(values["Voltage [V]"].data)
)

# Form dataset and locate the pulse
dataset = pybop.Dataset(
    {
        "Time [s]": values["Time [s]"].data,
        "Current function [A]": values["Current [A]"].data,
        "Discharge capacity [A.h]": values["Discharge capacity [A.h]"].data,
        "Voltage [V]": corrupt_values,
    }
)

# Define parameter set
parameter_set = convert_physical_to_electrode_parameters(
    model.parameter_set, "positive"
)

# Fitting parameters
parameters = pybop.Parameters(
    pybop.Parameter(
        "Particle diffusion time scale [s]",
        prior=pybop.Gaussian(2000, 1000),
    ),
    pybop.Parameter(
        "Series resistance [Ohm]",
        initial_value=parameter_set["Series resistance [Ohm]"],
    ),
)

# Define the cost to optimise
model = pybop.lithium_ion.SPDiffusion(parameter_set=parameter_set)
problem = pybop.FittingProblem(model, parameters, dataset)
cost = pybop.RootMeanSquaredError(problem)

# Build and run the optimisation problem
optim = pybop.SciPyMinimize(cost=cost)
results = optim.run()

# Plot the timeseries output
pybop.plot.quick(problem, problem_inputs=results.x, title="Optimised Comparison")

# Plot convergence
pybop.plot.convergence(optim)

# Plot the parameter traces
pybop.plot.parameters(optim)
