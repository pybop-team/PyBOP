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
parameter_values = pybamm.ParameterValues("Chen2020")

# Fix the electrolyte diffusivity and conductivity
ce0 = parameter_values["Initial concentration in electrolyte [mol.m-3]"]
T = parameter_values["Ambient temperature [K]"]
parameter_values["Electrolyte diffusivity [m2.s-1]"] = parameter_values[
    "Electrolyte diffusivity [m2.s-1]"
](ce0, T)
parameter_values["Electrolyte conductivity [S.m-1]"] = parameter_values[
    "Electrolyte conductivity [S.m-1]"
](ce0, T)

# Define a test protocol
init_soc = 0.9
experiment = pybamm.Experiment(
    [
        "Discharge at 1C until 2.5 V (5 seconds period)",
        "Rest for 30 minutes (5 seconds period)",
    ],
)

# Run an example SPMe simulation
model_options = {"surface form": "differential", "contact resistance": "true"}
grouped_parameter_values = pybop.lithium_ion.GroupedSPMe.create_grouped_parameters(
    parameter_values
)
SPMe_model = pybamm.lithium_ion.SPMe(options=model_options)
grouped_SPMe_model = pybop.lithium_ion.GroupedSPMe(options=model_options)
for model, param, line_style in zip(
    [SPMe_model, grouped_SPMe_model],
    [parameter_values, grouped_parameter_values],
    ["solid", "dash"],
    strict=False,
):
    solution = pybamm.Simulation(
        model, parameter_values=param, experiment=experiment
    ).solve(initial_soc=init_soc)
    dataset = pybop.Dataset(
        {
            "Time [s]": solution["Time [s]"].data,
            "Current function [A]": solution["Current [A]"].data,
            "Voltage [V]": solution["Voltage [V]"].data,
        }
    )
    plot_dict.add_traces(
        dataset["Time [s]"], dataset["Voltage [V]"], line_dash=line_style
    )
plot_dict()

# Set up figure
fig, ax = plt.subplots()
ax.grid()

# Compare models in the frequency domain
for model, param, line_style in zip(
    [SPMe_model, grouped_SPMe_model],
    [parameter_values, grouped_parameter_values],
    ["b", "r--"],
    strict=False,
):
    NSOC = 11
    Nfreq = 60
    fmin = 4e-4
    fmax = 1e3
    SOCs = np.linspace(0, 1, NSOC)
    frequencies = np.logspace(np.log10(fmin), np.log10(fmax), Nfreq)

    impedances = 1j * np.zeros((Nfreq, NSOC))
    for ii, SOC in enumerate(SOCs):
        param.set_initial_state(SOC)
        solution = pybop.pybamm.EISSimulator(
            model, parameter_values=param, f_eval=frequencies
        ).simulate()
        impedances[:, ii] = solution["Impedance"]
        ax.plot(np.real(impedances[:, ii]), -np.imag(impedances[:, ii]), line_style)

# Show figure
ax.set(xlabel=r"$Z_r(\omega)$ [$\Omega$]", ylabel=r"$-Z_j(\omega)$ [$\Omega$]")
ax.set_aspect("equal", "box")
plt.show()
