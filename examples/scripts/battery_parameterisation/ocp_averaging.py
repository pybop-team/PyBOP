import numpy as np
import pybamm
from pybamm import CasadiSolver

import pybop

# Generate some synthetic data for testing
model = pybamm.lithium_ion.SPMe()
parameter_values = pybamm.ParameterValues("Chen2020")

# Create representative charge and discharge datasets
parameter_values.set_initial_state(1.0)
discharge_solution = pybamm.Simulation(
    model,
    parameter_values=parameter_values,
    experiment=pybamm.Experiment(["Discharge at C/10 until 2.5 V"]),
    solver=CasadiSolver(),
).solve()
discharge_dataset_fullcell = pybop.Dataset(
    {
        "Stoichiometry": discharge_solution["Negative electrode stoichiometry"].data,
        "Voltage [V]": discharge_solution["Voltage [V]"].data,
    }
)
discharge_dataset_positive = pybop.Dataset(
    {
        "Stoichiometry": discharge_solution["Positive electrode stoichiometry"].data,
        "Voltage [V]": np.mean(
            discharge_solution[
                "Positive electrode surface potential difference [V]"
            ].data,
            axis=0,
        ),
    }
)
charge_dataset_negative = pybop.Dataset(
    {
        "Stoichiometry": discharge_solution["Negative electrode stoichiometry"].data,
        "Voltage [V]": np.mean(
            discharge_solution[
                "Negative electrode surface potential difference [V]"
            ].data,
            axis=0,
        ),
    }
)
parameter_values.set_initial_state(0.0)
charge_solution = pybamm.Simulation(
    model,
    parameter_values=parameter_values,
    experiment=pybamm.Experiment(["Charge at C/10 until 4.2 V"]),
    solver=CasadiSolver(),
).solve()
charge_dataset_fullcell = pybop.Dataset(
    {
        "Stoichiometry": charge_solution["Negative electrode stoichiometry"].data,
        "Voltage [V]": charge_solution["Voltage [V]"].data,
    }
)
charge_dataset_positive = pybop.Dataset(
    {
        "Stoichiometry": charge_solution["Positive electrode stoichiometry"].data,
        "Voltage [V]": np.mean(
            charge_solution["Positive electrode surface potential difference [V]"].data,
            axis=0,
        ),
    }
)
discharge_dataset_negative = pybop.Dataset(
    {
        "Stoichiometry": charge_solution["Negative electrode stoichiometry"].data,
        "Voltage [V]": np.mean(
            charge_solution["Negative electrode surface potential difference [V]"].data,
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
