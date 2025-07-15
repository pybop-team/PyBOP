from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pybamm
from scipy.io import savemat

import pybop

#
factor = 2
Nparams = 11
SOC = 0.5
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

grouped_parameters = pybop.lithium_ion.GroupedSPMe.apply_parameter_grouping(
    parameter_set
)
grouped_parameters["Series resistance [Ohm]"] = R0
model_options = {"surface form": "differential", "contact resistance": "true"}

var_pts = {"x_n": 100, "x_s": 20, "x_p": 100, "r_n": 100, "r_p": 100}

## Change parameters
parameter_name = "Negative electrode relative porosity"

# "Positive particle diffusion time scale [s]"
# "Positive electrode electrolyte diffusion time scale [s]"
# "Separator electrolyte diffusion time scale [s]"
# "Positive electrode charge transfer time scale [s]"
# "Series resistance [Ohm]"
# "Positive electrode relative porosity"
# "Cation transference number"
# "Reference electrolyte capacity [A.s]"
# "Positive electrode capacitance [F]"
# "Positive theoretical electrode capacity [As]"
# "Positive electrode relative thickness"
# "Measured cell capacity [A.s]"

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
        solver=pybamm.CasadiSolver(),
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
save_path = current_dir / "Data" / "Z_SPMegrouped_zetan.mat"
savemat(save_path, mdic)
