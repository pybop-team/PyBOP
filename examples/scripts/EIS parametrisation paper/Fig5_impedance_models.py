import numpy as np
import matplotlib.pyplot as plt
import pybop
import pybamm
from scipy.io import savemat


Nfreq = 60
SOC = 0.5
fmin = 2e-4
fmax = 1e3

frequencies = np.logspace(np.log10(fmin), np.log10(fmax), Nfreq)

impedances = 1j * np.zeros((Nfreq, 3))

# Define model
parameter_set = pybop.ParameterSet.pybamm("Chen2020")
parameter_set["Contact resistance [Ohm]"] = 0.01

model_options = {"surface form": "differential", "contact resistance": "true"}
var_pts = {"x_n": 100, "x_s": 20, "x_p": 100, "r_n": 100, "r_p": 100}


## SPM
model = pybop.lithium_ion.SPM(
    parameter_set=parameter_set, options=model_options, eis=True, var_pts=var_pts
)
model.build(initial_state={"Initial SoC": SOC})

simulation = model.simulateEIS(inputs=None, f_eval=frequencies)
impedances[:, 0] = simulation["Impedance"]

## SPMe
model = pybop.lithium_ion.SPMe(
    parameter_set=parameter_set, options=model_options, eis=True, var_pts=var_pts
)
model.build(initial_state={"Initial SoC": SOC})
simulation = model.simulateEIS(inputs=None, f_eval=frequencies)
impedances[:, 1] = simulation["Impedance"]

## DFN
model = pybop.lithium_ion.DFN(
    parameter_set=parameter_set, options=model_options, eis=True, var_pts=var_pts
)
model.build(initial_state={"Initial SoC": SOC})
simulation = model.simulateEIS(inputs=None, f_eval=frequencies)
impedances[:, 2] = simulation["Impedance"]

## Plot
fig, ax = plt.subplots()
for ii in range(3):
    ax.plot(
        np.real(impedances[:, ii]),
        -np.imag(impedances[:, ii]),
    )
ax.set(xlabel="$Z_r(\omega)$ [$\Omega$]", ylabel="$-Z_j(\omega)$ [$\Omega$]")
ax.grid()
ax.set_aspect("equal", "box")
plt.show()

## Save
mdic = {"Z": impedances, "f": frequencies}
savemat("Data/Z_SPM_SPMe_DFN_Pybop_chen2020.mat", mdic)
