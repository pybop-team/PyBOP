import numpy as np

import pybop

# Generate some synthetic data for testing
parameter_set = pybop.ParameterSet("Chen2020")
ocp_function = parameter_set["Positive electrode OCP [V]"]


def noise(sigma):
    return np.random.normal(0, sigma, 91)


discharge_sto = np.linspace(0, 0.9, 91)
discharge_voltage = ocp_function(discharge_sto + 0.02) + noise(1e-3)
charge_sto = np.linspace(1, 0.1, 91)
charge_voltage = ocp_function(charge_sto - 0.02) + noise(1e-3)

# Create the charge and discharge datasets
discharge_dataset = pybop.Dataset(
    {"Stoichiometry": discharge_sto, "Voltage [V]": discharge_voltage}
)
charge_dataset = pybop.Dataset(
    {"Stoichiometry": charge_sto, "Voltage [V]": charge_voltage}
)

# Estimate the shift and generate the average open-circuit potential
ocp_average = pybop.OCPAverage(
    discharge_dataset,
    charge_dataset,
    allow_stretching=False,
)

# Create a composite open-circuit potential from charge and discharge
ocp_blend = pybop.OCPBlend(discharge_dataset, charge_dataset)

# Verify the method through plotting
stoichiometry = np.linspace(0, 1, 101)
fig = pybop.plot.trajectories(
    x=[
        discharge_dataset["Stoichiometry"],
        charge_dataset["Stoichiometry"],
        stoichiometry,
        ocp_average.dataset["Stoichiometry"],
        ocp_blend.dataset["Stoichiometry"],
    ],
    y=[
        discharge_dataset["Voltage [V]"],
        charge_dataset["Voltage [V]"],
        parameter_set["Positive electrode OCP [V]"](stoichiometry),
        ocp_average.dataset["Voltage [V]"],
        ocp_blend.dataset["Voltage [V]"],
    ],
    trace_names=["Discharge", "Charge", "Ground truth", "Averaged", "Blended"],
)
