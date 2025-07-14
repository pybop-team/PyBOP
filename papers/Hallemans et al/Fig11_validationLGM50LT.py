from pathlib import Path

import matplotlib.pyplot as plt
import pybamm
import scipy
from scipy.io import savemat

import pybop

## Grouped parameter set
parameter_set = pybop.ParameterSet.pybamm("Chen2020")
parameter_set["Electrolyte diffusivity [m2.s-1]"] = 1.769e-10
parameter_set["Electrolyte conductivity [S.m-1]"] = 1e16
parameter_set["Negative electrode conductivity [S.m-1]"] = 1e16
parameter_set["Positive electrode conductivity [S.m-1]"] = 1e16

grouped_parameters = pybop.lithium_ion.GroupedSPMe.apply_parameter_grouping(parameter_set)

## Information battery About:Energy
current_dir = Path(__file__).parent
OCP_data_path = current_dir / "Data" / "LGM50LT" / "OCP_LGM50LT.mat"
OCP_data = scipy.io.loadmat(OCP_data_path)
x_pos_d = OCP_data.get("x_pos")
x_pos_d = x_pos_d.flatten()
x_neg_d = OCP_data.get("x_neg")
x_neg_d = x_neg_d.flatten()
U_pos_d = OCP_data.get("U_pos")
U_pos_d = U_pos_d.flatten()
U_neg_d = OCP_data.get("U_neg")
U_neg_d = U_neg_d.flatten()


def U_pos(sto):
    return pybamm.Interpolant(
        x_pos_d,
        U_pos_d,
        sto,
        name="U_pos_LGM50LT",
        interpolator="cubic",
        extrapolate=True,
    )


def U_neg(sto):
    return pybamm.Interpolant(
        x_neg_d,
        U_neg_d,
        sto,
        name="U_neg_LGM50LT",
        interpolator="cubic",
        extrapolate=True,
    )


L_pos = 55.6e-6
L_neg = 82.8e-6
L_sep = parameter_set["Separator thickness [m]"]
L_tot = L_pos + L_neg + L_sep
l_pos = L_pos / L_tot
l_neg = L_neg / L_tot

Q_meas = 3600 * 4.885  # [As]

grouped_parameters.update(
    {
        "Measured cell capacity [A.s]": Q_meas,
        "Positive electrode OCP [V]": U_pos,
        "Negative electrode OCP [V]": U_neg,
        "Positive electrode relative thickness": l_pos,
        "Negative electrode relative thickness": l_neg,
    },
    check_already_exists=False,
)

## Our parametrisation of the remaining parameters
params_path = current_dir / "Data" / "LGM50LT" / "Zhat3mV_SOC_SPMe_LGM50LT.mat"
params = scipy.io.loadmat(params_path)
thetahat = params.get("thetahat")
thetahat = thetahat.flatten()

parameters = pybop.Parameters(
    pybop.Parameter(
        "Series resistance [Ohm]",
    ),
    pybop.Parameter(
        "Positive particle diffusion time scale [s]",
    ),
    pybop.Parameter(
        "Negative particle diffusion time scale [s]",
    ),
    pybop.Parameter(
        "Cation transference number",
    ),
    pybop.Parameter(
        "Positive electrode electrolyte diffusion time scale [s]",
    ),
    pybop.Parameter(
        "Negative electrode electrolyte diffusion time scale [s]",
    ),
    pybop.Parameter(
        "Separator electrolyte diffusion time scale [s]",
    ),
    pybop.Parameter(
        "Positive electrode charge transfer time scale [s]",
    ),
    pybop.Parameter(
        "Negative electrode charge transfer time scale [s]",
    ),
    pybop.Parameter(
        "Positive electrode capacitance [F]",
    ),
    pybop.Parameter(
        "Negative electrode capacitance [F]",
    ),
    pybop.Parameter(
        "Positive electrode relative porosity",
    ),
    pybop.Parameter(
        "Negative electrode relative porosity",
    ),
    pybop.Parameter(
        "Reference electrolyte capacity [A.s]",
    ),
    pybop.Parameter(
        "Minimum positive stoichiometry",
    ),
    pybop.Parameter(
        "Maximum positive stoichiometry",
    ),
    pybop.Parameter(
        "Minimum negative stoichiometry",
    ),
    pybop.Parameter(
        "Maximum negative stoichiometry",
    ),
)

grouped_parameters.update(parameters.as_dict(thetahat))

print("Grouped parameters:", grouped_parameters)

## Create model
model_options = {"surface form": "differential", "contact resistance": "true"}
var_pts = {"x_n": 100, "x_s": 20, "x_p": 100, "r_n": 100, "r_p": 100}

model = pybop.lithium_ion.GroupedSPMe(
    parameter_set=grouped_parameters,
    var_pts=var_pts,
    options=model_options,
    solver=pybamm.CasadiSolver(dt_max=100),
)

## Load drive cycle
drivecycle_path = current_dir / "Data" / "LGM50LT" / "DrivecycleMeasurement_80to20.mat"
drivecycle = scipy.io.loadmat(drivecycle_path)
time = drivecycle.get("t")
current = drivecycle.get("i")
time = time.flatten()
current = -current.flatten()

## Validate model in the time domain for drive cycle
SOC0 = 0.8
model.build(initial_state={"Initial SoC": SOC0})

experiment = pybop.Dataset(
    {
        "Time [s]": time,
        "Current function [A]": current,
    }
)
model.set_current_function(dataset=experiment)
simulation = model.predict(t_eval=time)

dataset = pybop.Dataset(
    {
        "Time [s]": simulation["Time [s]"].data,
        "Current function [A]": simulation["Current [A]"].data,
        "Voltage [V]": simulation["Voltage [V]"].data,
    }
)

fig, ax = plt.subplots()
ax.plot(dataset["Time [s]"], dataset["Current function [A]"])
ax.set(xlabel="time [s]", ylabel="Current [A]")
ax.grid()
plt.show()

fig, ax = plt.subplots()
ax.plot(dataset["Time [s]"], dataset["Voltage [V]"])
ax.set(xlabel="time [s]", ylabel="Voltage [V]")
ax.grid()
plt.show()

## Save data
t = dataset["Time [s]"].data
i = dataset["Current function [A]"].data
v = dataset["Voltage [V]"].data

mdic = {"t": t, "i": i, "v": v, "SOC0": SOC0}
save_path = current_dir / "Data" / "Validation_SPMegrouped80to20.mat"
savemat(save_path, mdic)
