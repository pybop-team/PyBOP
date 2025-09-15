import matplotlib.pyplot as plt
import numpy as np
import pybamm

import pybop

# Prepare figure
layout_options = dict(
    xaxis_title="Time / s",
    yaxis_title="Voltage / V",
)
plot_dict = pybop.plot.StandardPlot(layout_options=layout_options)

# Unpack parameter values from Chen2020
parameter_set = pybamm.ParameterValues("Chen2020")

# Fix the electrolyte diffusivity and conductivity
ce0 = parameter_set["Initial concentration in electrolyte [mol.m-3]"]
T = parameter_set["Ambient temperature [K]"]
parameter_set["Electrolyte diffusivity [m2.s-1]"] = parameter_set[
    "Electrolyte diffusivity [m2.s-1]"
](ce0, T)
parameter_set["Electrolyte conductivity [S.m-1]"] = parameter_set[
    "Electrolyte conductivity [S.m-1]"
](ce0, T)

# Define a test protocol
initial_state = {"Initial SoC": 0.9}
experiment = pybamm.Experiment(
    [
        "Discharge at 1C until 2.5 V (5 seconds period)",
        "Rest for 30 minutes (5 seconds period)",
    ],
)

# Run an example SPMe simulation
model_options = {"surface form": "differential", "contact resistance": "true"}
time_domain_SPMe = pybop.lithium_ion.SPMe(
    parameter_set=parameter_set,
    options=model_options,
)
time_domain_SPMe.solver = time_domain_SPMe.pybamm_model.default_solver
simulation = time_domain_SPMe.predict(
    initial_state=initial_state, experiment=experiment
)
dataset = pybop.Dataset(
    {
        "Time [s]": simulation["Time [s]"].data,
        "Current function [A]": simulation["Current [A]"].data,
        "Voltage [V]": simulation["Voltage [V]"].data,
    }
)
plot_dict.add_traces(dataset["Time [s]"], dataset["Voltage [V]"])

# Test model in the time domain
grouped_parameter_set = pybop.lithium_ion.GroupedSPMe.apply_parameter_grouping(
    parameter_set
)
time_domain_grouped = pybop.lithium_ion.GroupedSPMe(
    parameter_set=grouped_parameter_set,
    options=model_options,
    build=True,
)
time_domain_grouped.set_initial_state(initial_state)
time_domain_grouped.set_current_function(dataset)
simulation = time_domain_grouped.predict(t_eval=dataset["Time [s]"])
dataset = pybop.Dataset(
    {
        "Time [s]": simulation["Time [s]"].data,
        "Current function [A]": simulation["Current [A]"].data,
        "Voltage [V]": simulation["Voltage [V]"].data,
    }
)
plot_dict.add_traces(dataset["Time [s]"], dataset["Voltage [V]"], line_dash="dash")
plot_dict()

# Set up figure
fig, ax = plt.subplots()
ax.grid()

# Compare models in the frequency domain
freq_domain_SPMe = pybop.lithium_ion.SPMe(
    parameter_set=parameter_set, options=model_options, eis=True
)
freq_domain_grouped = pybop.lithium_ion.GroupedSPMe(
    parameter_set=grouped_parameter_set,
    options=model_options,
    eis=True,
    build=True,
)

for i, model in enumerate([freq_domain_SPMe, freq_domain_grouped]):
    NSOC = 11
    Nfreq = 60
    fmin = 4e-4
    fmax = 1e3
    SOCs = np.linspace(0, 1, NSOC)
    frequencies = np.logspace(np.log10(fmin), np.log10(fmax), Nfreq)

    impedances = 1j * np.zeros((Nfreq, NSOC))
    for ii, SOC in enumerate(SOCs):
        model.set_initial_state({"Initial SoC": SOC})
        simulation = model.simulateEIS(inputs=None, f_eval=frequencies)
        impedances[:, ii] = simulation["Impedance"]

        if i == 0:
            ax.plot(np.real(impedances[:, ii]), -np.imag(impedances[:, ii]), "b")
        if i == 1:
            ax.plot(np.real(impedances[:, ii]), -np.imag(impedances[:, ii]), "r--")

# Show figure
ax.set(xlabel=r"$Z_r(\omega)$ [$\Omega$]", ylabel=r"$-Z_j(\omega)$ [$\Omega$]")
ax.set_aspect("equal", "box")
plt.show()
