import numpy as np
import pybamm

import pybop

"""
An example to demonstrate the functionality of `pybop.OCPAverage`.
"""

# Generate some synthetic data for testing
model = pybamm.lithium_ion.SPMe()
parameter_values = pybamm.ParameterValues("Chen2020")

# Create representative charge and discharge datasets
discharge_sol = pybamm.Simulation(
    model,
    parameter_values=parameter_values,
    experiment=pybamm.Experiment(["Discharge at C/10 until 2.5 V"]),
).solve(initial_soc=1.0)
t = np.linspace(0, discharge_sol.t[-1], 101)
discharge_dataset_fullcell = pybop.Dataset(
    {
        "Stoichiometry": discharge_sol["Negative electrode stoichiometry"](t),
        "Voltage [V]": discharge_sol["Voltage [V]"](t),
    }
)
discharge_dataset_positive = pybop.Dataset(
    {
        "Stoichiometry": discharge_sol["Positive electrode stoichiometry"](t),
        "Voltage [V]": np.mean(
            discharge_sol["Positive electrode surface potential difference [V]"](t),
            axis=0,
        ),
    }
)
charge_dataset_negative = pybop.Dataset(
    {
        "Stoichiometry": discharge_sol["Negative electrode stoichiometry"](t),
        "Voltage [V]": np.mean(
            discharge_sol["Negative electrode surface potential difference [V]"](t),
            axis=0,
        ),
    }
)
charge_sol = pybamm.Simulation(
    model,
    parameter_values=parameter_values,
    experiment=pybamm.Experiment(["Charge at C/10 until 4.2 V"]),
).solve(initial_soc=0.0)
t = np.linspace(0, charge_sol.t[-1], 101)
charge_dataset_fullcell = pybop.Dataset(
    {
        "Stoichiometry": charge_sol["Negative electrode stoichiometry"](t),
        "Voltage [V]": charge_sol["Voltage [V]"](t),
    }
)
charge_dataset_positive = pybop.Dataset(
    {
        "Stoichiometry": charge_sol["Positive electrode stoichiometry"](t),
        "Voltage [V]": np.mean(
            charge_sol["Positive electrode surface potential difference [V]"](t),
            axis=0,
        ),
    }
)
discharge_dataset_negative = pybop.Dataset(
    {
        "Stoichiometry": charge_sol["Negative electrode stoichiometry"](t),
        "Voltage [V]": np.mean(
            charge_sol["Negative electrode surface potential difference [V]"](t),
            axis=0,
        ),
    }
)


for discharge_dataset, charge_dataset in zip(
    [
        discharge_dataset_fullcell,
        discharge_dataset_negative,
        discharge_dataset_positive,
    ],
    [charge_dataset_fullcell, charge_dataset_negative, charge_dataset_positive],
    strict=False,
):
    # Estimate the shift and generate the average open-circuit potential
    ocp_average = pybop.OCPAverage(
        discharge_dataset,
        charge_dataset,
        allow_stretching=False,
    )
    average_dataset = ocp_average()

    # Verify the method through plotting
    stoichiometry = np.linspace(0, 1, 101)
    stos = [
        discharge_dataset["Stoichiometry"],
        charge_dataset["Stoichiometry"],
        average_dataset["Stoichiometry"],
    ]
    volt = [
        discharge_dataset["Voltage [V]"],
        charge_dataset["Voltage [V]"],
        average_dataset["Voltage [V]"],
    ]
    trace_names = ["Discharge", "Charge", "Averaged"]
    legend = dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    fig = pybop.plot.trajectories(
        x=stos,
        y=volt,
        trace_names=trace_names,
        xaxis_title="Stoichiometry",
        yaxis_title="Voltage [V]",
        legend=legend,
    )

    dcap = [
        np.gradient(
            discharge_dataset["Stoichiometry"], discharge_dataset["Voltage [V]"]
        ),
        np.gradient(charge_dataset["Stoichiometry"], charge_dataset["Voltage [V]"]),
        np.gradient(average_dataset["Stoichiometry"], average_dataset["Voltage [V]"]),
    ]
    fig = pybop.plot.trajectories(
        x=stos,
        y=dcap,
        trace_names=trace_names,
        xaxis_title="Stoichiometry",
        yaxis_title="Differential capacity [V-1]",
        legend=legend,
    )
