import matplotlib.pyplot as plt
import numpy as np
import pybamm

import pybop

"""
Example demonstrating EIS applied during operation (slow dis/charge) of the cell.
"""

# Define model and parameter values
model = pybamm.lithium_ion.SPMe(
    options={"surface form": "differential", "contact resistance": "true"}
)
parameter_values = pybamm.ParameterValues("Chen2020")
parameter_values["Contact resistance [Ohm]"] = 0.02
parameter_values.set_initial_state("2.85 V", options=model.options)

# Set up and run a charge/discharge experiment
C_rate = parameter_values["Nominal cell capacity [A.h]"]
dataset = pybop.Dataset(
    {
        "Time [s]": np.asarray(
            [0, 1, 1001, 2001, 3001, 3002, 3003, 4003, 5003, 6003, 6004]
        ),
        "Current function [A]": np.asarray([0, -1, -1, -1, -1, 0, 1, 1, 1, 1, 0])
        * C_rate
        / 3,
    }
)

sim = pybop.pybamm.Simulator(
    model, parameter_values=parameter_values, protocol=dataset
)
solution = sim.solve()
solution.plot()

# Set up and run the simulation
n_frequency = 60
solution = pybop.pybamm.EISSimulator(
    model,
    parameter_values=parameter_values,
    f_eval=np.logspace(-4, 5, n_frequency),
    protocol=dataset,
).solve()

fig, ax = plt.subplots()
n_time_steps = len(solution["Time [s]"].data)
for i in range(n_time_steps):
    impedance = solution["Impedance"].data[i, :]
    ax.plot(
        np.real(impedance),
        -np.imag(impedance),
        "-" if i < n_time_steps / 2 else "--",
        label=f"t={solution['Time [s]'].data[i]}s",
    )
ax.set(xlabel=r"$Z_r(\omega)$ [$\Omega$]", ylabel=r"$-Z_j(\omega)$ [$\Omega$]")
ax.set_aspect("equal", "box")
ax.legend()
ax.set_ylim([0, ax.get_xlim()[1]])
plt.show()
