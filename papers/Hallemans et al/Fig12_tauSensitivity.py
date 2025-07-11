from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat

import pybop
from pybop.models.lithium_ion.basic_SPMe import BaseGroupedSPMe

SOC = 0.2

factor = 2
Nparams = 11
Nfreq = 60
fmin = 2e-4
fmax = 1e3

# Get grouped parameters
R0 = 0.01

parameter_set = pybop.ParameterSet.pybamm("Chen2020")
parameter_set["Electrolyte diffusivity [m2.s-1]"] = 1.769e-10
parameter_set["Electrolyte conductivity [S.m-1]"] = 1e16
parameter_set["Negative electrode conductivity [S.m-1]"] = 1e16
parameter_set["Positive electrode conductivity [S.m-1]"] = 1e16

grouped_parameters = BaseGroupedSPMe.apply_parameter_grouping(parameter_set)
grouped_parameters["Series resistance [Ohm]"] = R0
model_options = {"surface form": "differential", "contact resistance": "true"}
var_pts = {"x_n": 100, "x_s": 20, "x_p": 100, "r_n": 100, "r_p": 100}

## Change parameters
parameter_name = "Negative particle diffusion time scale [s]"
param0 = grouped_parameters[parameter_name]
params = np.logspace(np.log10(param0 / factor), np.log10(param0 * factor), Nparams)

# Simulate impedance at these parameter values
frequencies = np.logspace(np.log10(fmin), np.log10(fmax), Nfreq)

impedances = 1j * np.zeros((Nfreq, Nparams))
for ii, param in enumerate(params):
    grouped_parameters[parameter_name] = param
    model = pybop.lithium_ion.GroupedSPMe(
        parameter_set=grouped_parameters,
        eis=True,
        options=model_options,
        var_pts=var_pts,
    )
    model.build(
        initial_state={"Initial SoC": SOC},
    )
    simulation = model.simulateEIS(inputs=None, f_eval=frequencies)
    impedances[:, ii] = simulation["Impedance"]

fig, ax = plt.subplots()
for ii in range(Nparams):
    ax.plot(
        np.real(impedances[:, ii]),
        -np.imag(impedances[:, ii]),
    )
ax.set(xlabel=r"$Z_r(\omega)$ [$\Omega$]", ylabel=r"$-Z_j(\omega)$ [$\Omega$]")
ax.grid()
ax.set_aspect("equal", "box")
plt.show()

mdic = {"Z": impedances, "f": frequencies, "name": parameter_name}
current_dir = Path(__file__).parent
save_path = current_dir / "Data" / "Z_SPMegrouped_taudn_20.mat"
savemat(save_path, mdic)
