import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat

import pybop
from pybop.models.lithium_ion.basic_SPMe import convert_physical_to_grouped_parameters

## Grouped parameter set
R0 = 0.01

parameter_set = pybop.ParameterSet.pybamm("Chen2020")
parameter_set["Electrolyte diffusivity [m2.s-1]"] = 1.769e-10
parameter_set["Electrolyte conductivity [S.m-1]"] = 1e16
parameter_set["Negative electrode conductivity [S.m-1]"] = 1e16
parameter_set["Positive electrode conductivity [S.m-1]"] = 1e16

grouped_parameters = convert_physical_to_grouped_parameters(parameter_set)
grouped_parameters["Series resistance [Ohm]"] = R0
model_options = {"surface form": "differential", "contact resistance": "true"}
var_pts = {"x_n": 100, "x_s": 20, "x_p": 100, "r_n": 100, "r_p": 100}

## Create model
model = pybop.lithium_ion.GroupedSPMe(
    parameter_set=grouped_parameters, eis=True, var_pts=var_pts, options=model_options
)

## Test model in the time domain
SOC0 = 0.9
model.build(initial_state={"Initial SoC": SOC0})

Ts = 10  # Sampling period
T = 2 * 60 * 60  # 2h
N = int(T / Ts)
time = np.linspace(0, T - Ts, N)

i_relax7 = np.zeros([7 * int(60 / Ts)])  # 7 min
i_relax20 = np.zeros([20 * int(60 / Ts)])  # 20 min
i_discharge = 5 * np.ones([53 * int(60 / Ts)])  # 53 min
i_charge = -5 * np.ones([20 * int(60 / Ts)])  # 20 min
current = np.concatenate((i_relax7, i_discharge, i_relax20, i_charge, i_relax20))

experiment = pybop.Dataset(
    {
        "Time [s]": time,
        "Current function [A]": current,
    }
)
model.set_current_function(dataset=experiment)
simulation = model.predict(t_eval=time)

# Plot traces
fig, ax = plt.subplots()
ax.plot(simulation["Time [s]"].data, simulation["Current [A]"].data)
ax.set(xlabel="time [s]", ylabel="Current [A]")
ax.grid()
plt.show()

fig, ax = plt.subplots()
ax.plot(simulation["Time [s]"].data, simulation["Voltage [V]"].data)
ax.set(xlabel="time [s]", ylabel="Voltage [V]")
ax.grid()
plt.show()

## Save data
t = simulation["Time [s]"].data
i = simulation["Current [A]"].data
v = simulation["Voltage [V]"].data

mdic = {"t": t, "i": i, "v": v, "SOC0": SOC0}
savemat("Data/timeDomainSimulation_SPMegrouped.mat", mdic)
