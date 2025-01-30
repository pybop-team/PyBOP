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

# Fit the GITT pulse using the single particle diffusion model
gitt_fit = pybop.gitt_pulse_fit(dataset, parameter_set)

# Plot the timeseries output
pybop.plot.quick(
    gitt_fit.problem, problem_inputs=gitt_fit.results.x, title="Optimised Comparison"
)

# Plot convergence
pybop.plot.convergence(gitt_fit.optim)

# Plot the parameter traces
pybop.plot.parameters(gitt_fit.optim)
