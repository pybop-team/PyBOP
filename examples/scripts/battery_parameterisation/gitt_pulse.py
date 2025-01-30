import numpy as np

import pybop
from pybop.models.lithium_ion.weppner_huggins import (
    convert_physical_to_electrode_parameters,
)

# Define model
parameter_set = pybop.ParameterSet("Xu2019")
model = pybop.lithium_ion.SPM(
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
pulse_index = np.flatnonzero(dataset["Current function [A]"])

# Define parameter set
parameter_set = convert_physical_to_electrode_parameters(parameter_set, "positive")
parameter_set.update(
    {
        "Reference voltage [V]": dataset["Voltage [V]"][pulse_index[0]],
        "Derivative of the OCP wrt stoichiometry [V]": (
            (dataset["Voltage [V]"][-1] - dataset["Voltage [V]"][0])
            / (
                (
                    dataset["Discharge capacity [A.h]"][-1]
                    - dataset["Discharge capacity [A.h]"][0]
                )
                / (parameter_set["Theoretical electrode capacity [A.s]"] / 3600)
            )
        ),
    },
)

# Fitting parameters
parameters = pybop.Parameters(
    pybop.Parameter(
        "Particle diffusion time scale [s]",
        prior=pybop.Gaussian(6000, 1000),
        bounds=[2000, 10000],
    ),
    pybop.Parameter(
        "Reference voltage [V]",
        initial_value=parameter_set["Reference voltage [V]"],
    ),
)

# Define the cost to optimise
model = pybop.lithium_ion.WeppnerHuggins(parameter_set=parameter_set)
dataset = dataset.get_subset(pulse_index)
problem = pybop.FittingProblem(model, parameters, dataset)
cost = pybop.RootMeanSquaredError(problem)

# Build the optimisation problem
optim = pybop.SciPyMinimize(cost=cost)

# Run the optimisation problem
results = optim.run()

# Plot the timeseries output
pybop.plot.quick(problem, problem_inputs=results.x, title="Optimised Comparison")

# Plot convergence
pybop.plot.convergence(optim)

# Plot the parameter traces
pybop.plot.parameters(optim)
