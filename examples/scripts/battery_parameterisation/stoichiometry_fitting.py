import numpy as np
import pybamm

import pybop

# Generate some synthetic data for testing
parameter_set = pybamm.ParameterValues("Chen2020")
ocv_function = parameter_set["Positive electrode OCP [V]"]
nom_capacity = parameter_set["Nominal cell capacity [A.h]"]


def noise(sigma):
    return np.random.normal(0, sigma, 91)


sto = np.linspace(0, 0.9, 91)
voltage = ocv_function(sto) + noise(2e-3)

# Create the OCV dataset
ocv_dataset = pybop.Dataset(
    {"Charge capacity [A.h]": (sto + 0.1) * nom_capacity, "Voltage [V]": voltage}
)

# Estimate the stoichiometry corresponding to the GITT-OCV
ocv_fit = pybop.OCPCapacityToStoichiometry(ocv_dataset, ocv_function)
fitted_dataset = ocv_fit()

# Verify the method through plotting
stoichiometry = np.linspace(0, 1, 101)
fig = pybop.plot.trajectories(
    x=[stoichiometry, fitted_dataset["Stoichiometry"]],
    y=[
        parameter_set["Positive electrode OCP [V]"](stoichiometry),
        fitted_dataset["Voltage [V]"],
    ],
    trace_names=["Ground truth", "Data vs. stoichiometry"],
)
