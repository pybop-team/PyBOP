import matplotlib.pyplot as plt
import numpy as np
import pybamm

import pybop

"""
This example introduces the grouped models, i.e. which group the correlated parameters
of the corresponding lithium-ion cell models into a minimum set of parameters for
identification purposes.
"""

# Prepare figure
fig = plt.figure()
plt.xlabel("Time / s")
plt.ylabel("Voltage / V")

# Use the Chen2020 parameters
parameter_values = pybamm.ParameterValues("Chen2020")

# Fix the electrolyte conductivity
ce0 = parameter_values["Initial concentration in electrolyte [mol.m-3]"]
T = parameter_values["Ambient temperature [K]"]
parameter_values["Electrolyte conductivity [S.m-1]"] = parameter_values[
    "Electrolyte conductivity [S.m-1]"
](ce0, T)

# Define the models to compare
model_options = {"surface form": "differential", "contact resistance": "true"}
models = [
    pybamm.lithium_ion.DFN(options=model_options),
    pybop.lithium_ion.GroupedDFN(options=model_options),
    pybop.lithium_ion.GroupedSPMe(options=model_options),
    pybop.lithium_ion.GroupedSPM(options=model_options),
]
params = [
    parameter_values,
    pybop.lithium_ion.GroupedDFN.create_grouped_parameters(parameter_values),
    pybop.lithium_ion.GroupedSPMe.create_grouped_parameters(parameter_values),
    pybop.lithium_ion.GroupedSPM.create_grouped_parameters(parameter_values),
]
linestyles = ["b", "c--", "y--", "m--"]

# Compare models in the time domain
init_soc = 0.9
experiment = pybamm.Experiment(
    [
        "Discharge at 2C until 2.5 V (5 seconds period)",
        "Rest for 20 minutes (5 seconds period)",
    ],
)
for model, param, linestyle in zip(models, params, linestyles, strict=False):
    solution = pybamm.Simulation(
        model, parameter_values=param, experiment=experiment, cache_esoh=False
    ).solve(initial_soc=init_soc)
    dataset = pybop.import_pybamm_solution(solution)
    plt.plot(dataset["Time [s]"], dataset["Voltage [V]"], linestyle, label=model.name)
plt.legend()

# Prepare figure
fig, ax = plt.subplots()
ax.grid()

# Compare models in the frequency domain
NSOC = 9
Nfreq = 60
fmin = 4e-4
fmax = 1e3
SOCs = np.linspace(0.1, 0.9, NSOC)
frequencies = np.logspace(np.log10(fmin), np.log10(fmax), Nfreq)
for model, param, linestyle in zip(models, params, linestyles, strict=False):
    for i, SOC in enumerate(SOCs):
        param.set_initial_state(SOC)
        solution = pybop.pybamm.EISSimulator(
            model, parameter_values=param, f_eval=frequencies
        ).solve()
        ax.plot(
            solution["Impedance [Ohm]"].data.real,
            -solution["Impedance [Ohm]"].data.imag,
            linestyle,
            label=model.name if i == 0 else None,
        )

# Show figure
ax.set(xlabel=r"$Z_r(\omega)$ [$\Omega$]", ylabel=r"$-Z_j(\omega)$ [$\Omega$]")
ax.set_aspect("equal", "box")
ax.legend()
plt.show()
