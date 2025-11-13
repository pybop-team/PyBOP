import numpy as np
import pandas as pd
import pybamm

# Define model and use high-performant solver for sensitivities
solver = pybamm.CasadiSolver(atol=1e-7, rtol=1e-7)
parameter_values = pybamm.ParameterValues("Chen2020")
models = [
    (pybamm.lithium_ion.DFN(parameter_values=parameter_values, solver=solver), "dfn"),
    (pybamm.lithium_ion.SPMe(parameter_values=parameter_values, solver=solver), "spme"),
    (pybamm.lithium_ion.SPM(parameter_values=parameter_values, solver=solver), "spm"),
]

# Generate data
sigma = 5e-4
soc = 0.75
experiment = pybamm.Experiment(
    [
        "Rest for 2 seconds (1 second period)",
        "Discharge at 0.5C for 40 minutes (20 second period)",
        "Charge at 0.5C for 20 minutes (20 second period)",
    ]
)
for model, name in models:
    values = model.predict(
        initial_state={"Initial SoC": soc},
        experiment=experiment,
        parameter_values=parameter_values,
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
        f"{name}_charge_discharge_{int(soc * 100)}.csv", index=False
    )
