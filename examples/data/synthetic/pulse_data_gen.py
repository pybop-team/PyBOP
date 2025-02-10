import numpy as np
import pandas as pd
import pybamm

import pybop

# Define model and use high-performant solver for sensitivities
solver = pybamm.CasadiSolver(atol=1e-7, rtol=1e-7)
parameter_set = pybop.ParameterSet("Chen2020")
models = [
    (pybop.lithium_ion.DFN(parameter_set=parameter_set, solver=solver), "dfn"),
    (pybop.lithium_ion.SPMe(parameter_set=parameter_set, solver=solver), "spme"),
    (pybop.lithium_ion.SPM(parameter_set=parameter_set, solver=solver), "spm"),
]

# Generate data
sigma = 2e-4
soc = 0.05
experiment = pybop.Experiment(
    [
        "Rest for 2 seconds (1 second period)",
        "Discharge at 0.1C for 1 minute (2 second period)",
        "Rest for 10 minutes (8 second period)",
        "Charge at 0.1C for 1 minute (2 second period)",
        "Rest for 10 minutes (8 second period)",
    ]
)
for model, name in models:
    values = model.predict(
        initial_state={"Initial SoC": soc},
        experiment=experiment,
        parameter_set=parameter_set,
    )

    def noise(sigma, dict_obj):
        return np.random.normal(0, sigma, len(dict_obj["Voltage [V]"].data))

    pd.DataFrame(
        {
            "Time [s]": np.round(values["Time [s]"].data, decimals=5),
            "Current function [A]": values["Current [A]"].data,
            "Voltage [V]": values["Voltage [V]"].data + noise(sigma, values),
            "Bulk open-circuit voltage [V]": values[
                "Bulk open-circuit voltage [V]"
            ].data
            + noise(sigma, values),
        }
    ).drop_duplicates(subset=["Time [s]"]).to_csv(
        f"{name}_pulse_{int(soc * 100)}.csv", index=False
    )
