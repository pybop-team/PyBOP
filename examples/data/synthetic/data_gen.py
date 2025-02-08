import numpy as np
import pandas as pd
import pybamm

import pybop

# Define model and use high-performant solver for sensitivities
solver = pybamm.CasadiSolver()
parameter_set = pybop.ParameterSet("Chen2020")
model = pybop.lithium_ion.DFN(parameter_set=parameter_set, solver=solver)

# Generate data
sigma = 0.002
experiment = pybop.Experiment(
    [
        "Rest for 2 seconds (1 second period)",
        "Discharge at 0.5C for 30 minutes (20 second period)",
        "Charge at 0.5C for 30 minutes (20 second period)",
    ]
)
values = model.predict(
    initial_state={"Initial SoC": 0.75},
    experiment=experiment,
    parameter_set=parameter_set,
)


def noise(sigma):
    return np.random.normal(0, sigma, len(values["Voltage [V]"].data))


pd.DataFrame(
    {
        "Time [s]": np.round(values["Time [s]"].data, decimals=5),
        "Current function [A]": values["Current [A]"].data,
        "Voltage [V]": values["Voltage [V]"].data + noise(sigma),
        "Bulk open-circuit voltage [V]": values["Bulk open-circuit voltage [V]"].data
        + noise(sigma),
    }
).drop_duplicates(subset=["Time [s]"]).to_csv("charge_discharge_75.csv", index=False)
