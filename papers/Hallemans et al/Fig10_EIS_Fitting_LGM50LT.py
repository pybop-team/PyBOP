import numpy as np
import scipy
import pybop
import pybamm
import pickle
import matplotlib.pyplot as plt
import time as timer
from scipy.io import savemat
from pybop.models.lithium_ion.basic_SPMe import convert_physical_to_grouped_parameters


## Parameter set

parameter_set = pybop.ParameterSet.pybamm("Chen2020")
parameter_set["Electrolyte diffusivity [m2.s-1]"] = 1.769e-10
parameter_set["Electrolyte conductivity [S.m-1]"] = 1e16  # 0.9487
parameter_set["Negative electrode conductivity [S.m-1]"] = 1e16
parameter_set["Positive electrode conductivity [S.m-1]"] = 1e16

grouped_parameters = convert_physical_to_grouped_parameters(parameter_set)

## Information battery About:Energy
OCP_data = scipy.io.loadmat("Data/LGM50LT/OCP_LGM50LT.mat")
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

model_options = {"surface form": "differential", "contact resistance": "true"}

var_pts = {"x_n": 100, "x_s": 20, "x_p": 100, "r_n": 100, "r_p": 100}

## Model
model = pybop.lithium_ion.GroupedSPMe(
    parameter_set=grouped_parameters, eis=True, var_pts=var_pts, options=model_options
)

## Choose parameter bounds for optimisation
tau_d_bounds = [5e2, 1e4]
tau_e_bounds = [1e2, 1e3]
tau_ct_bounds = [1e1, 5e3]
zeta_bounds = [0.5, 1.7]
Qe_bounds = [5e2, 2e3]
C_bounds = [0, 2]
c0p_bounds = [0.982, 0.984]
c0n_bounds = [0.0014, 0.0015]
c100p_bounds = [0.1897, 0.1899]
c100n_bounds = [0.8862, 0.8864]
# x_0_pos = 0.983
# x_0_neg = 0.00144
# x_100_pos = 0.1898
# x_100_neg = 0.88632
t_plus_bounds = [0.2, 0.7]
R0_bounds = [0.0125, 0.016]

parameters = pybop.Parameters(
    pybop.Parameter(
        "Series resistance [Ohm]",
        bounds=R0_bounds,
        initial_value=np.mean(R0_bounds),
    ),
    pybop.Parameter(
        "Positive particle diffusion time scale [s]",
        bounds=tau_d_bounds,
        initial_value=np.mean(tau_d_bounds),
    ),
    pybop.Parameter(
        "Negative particle diffusion time scale [s]",
        bounds=tau_d_bounds,
        initial_value=np.mean(tau_d_bounds),
    ),
    pybop.Parameter(
        "Cation transference number",
        bounds=np.sort(t_plus_bounds),
        initial_value=np.mean(t_plus_bounds),
    ),
    pybop.Parameter(
        "Positive electrode electrolyte diffusion time scale [s]",
        bounds=np.sort(tau_e_bounds),
        initial_value=np.mean(tau_e_bounds),
    ),
    pybop.Parameter(
        "Negative electrode electrolyte diffusion time scale [s]",
        bounds=np.sort(tau_e_bounds),
        initial_value=np.mean(tau_e_bounds),
    ),
    pybop.Parameter(
        "Separator electrolyte diffusion time scale [s]",
        bounds=np.sort(tau_e_bounds),
        initial_value=np.mean(tau_e_bounds),
    ),
    pybop.Parameter(
        "Positive electrode charge transfer time scale [s]",
        bounds=np.sort(tau_ct_bounds),
        initial_value=np.mean(tau_ct_bounds),
    ),
    pybop.Parameter(
        "Negative electrode charge transfer time scale [s]",
        bounds=np.sort(tau_ct_bounds),
        initial_value=np.mean(tau_ct_bounds),
    ),
    pybop.Parameter(
        "Positive electrode capacitance [F]",
        bounds=np.sort(C_bounds),
        initial_value=np.mean(C_bounds),
    ),
    pybop.Parameter(
        "Negative electrode capacitance [F]",
        bounds=np.sort(C_bounds),
        initial_value=np.mean(C_bounds),
    ),
    pybop.Parameter(
        "Positive electrode relative porosity",
        bounds=np.sort(zeta_bounds),
        initial_value=np.mean(zeta_bounds),
    ),
    pybop.Parameter(
        "Negative electrode relative porosity",
        bounds=np.sort(zeta_bounds),
        initial_value=np.mean(zeta_bounds),
    ),
    pybop.Parameter(
        "Reference electrolyte capacity [A.s]",
        bounds=np.sort(Qe_bounds),
        initial_value=np.mean(Qe_bounds),
    ),
    pybop.Parameter(
        "Minimum positive stoichiometry",
        bounds=np.sort(c100p_bounds),
        initial_value=np.mean(c100p_bounds),
    ),
    pybop.Parameter(
        "Maximum positive stoichiometry",
        bounds=np.sort(c0p_bounds),
        initial_value=np.mean(c0p_bounds),
    ),
    pybop.Parameter(
        "Minimum negative stoichiometry",
        bounds=np.sort(c0n_bounds),
        initial_value=np.mean(c0n_bounds),
    ),
    pybop.Parameter(
        "Maximum negative stoichiometry",
        bounds=np.sort(c100n_bounds),
        initial_value=np.mean(c100n_bounds),
    ),
)


## Read impedance data LG M50LT
EIS_data = scipy.io.loadmat("Data/LGM50LT/impedanceLGM50LT_Hybrid_4h_3mVrms.mat")

impedances = EIS_data.get("Z")
frequencies = EIS_data.get("f")
frequencies = frequencies.flatten()
SOCs = EIS_data.get("SOC") / 100
SOCs = SOCs.flatten()
OCVs = EIS_data.get("OCV")
OCVs = OCVs.flatten()

# Remove 10% and 90% SOC impedance (quite different from the model)
SOCs = SOCs[1:8]
OCVs = OCVs[1:8]
impedances = impedances[:, 1:8]

## Create models and datasets

# Form dataset
signal = ["Impedance"]
datasets = [None] * len(SOCs)
models = [None] * len(SOCs)
problems = [None] * len(SOCs)

for ii in range(len(SOCs)):
    datasets[ii] = pybop.Dataset(
        {
            "Frequency [Hz]": frequencies,
            "Current function [A]": np.ones(len(frequencies)) * 0.0,
            "Impedance": impedances[:, ii],
        }
    )
    models[ii] = model.new_copy()
    models[ii].build(initial_state={"Initial SoC": SOCs[ii]})
    problems[ii] = pybop.FittingProblem(
        models[ii],
        parameters,
        datasets[ii],
        signal=signal,
    )


problem = pybop.MultiFittingProblem(
    problems[0],
    problems[1],
    problems[2],
    problems[3],
    problems[4],
    problems[5],
    problems[6],
)

cost = pybop.SumSquaredError(problem)
optim = pybop.PSO(cost, max_iterations=2000, max_unchanged_iterations=1000)

# Run optimisation
start_time = timer.time()
estimate = optim.run()
thetahat = estimate.x
end_time = timer.time()
computationTime = end_time - start_time

# Print optimised parameters
optimised_grouped_parameters = grouped_parameters
optimised_grouped_parameters.update(parameters.as_dict(thetahat))

print("Optimised grouped parameters:", optimised_grouped_parameters)

# Plot convergence
pybop.plot.convergence(optim)

# Plot the parameter traces
pybop.plot.parameters(optim)

## Save estimated impedance data
modelhat = pybop.lithium_ion.GroupedSPMe(
    parameter_set=optimised_grouped_parameters,
    eis=True,
    var_pts=var_pts,
    options=model_options,
)

Nfreq = len(frequencies)
NSOC = len(SOCs)

# Compute impedance for estimated parameters
impedanceshat = 1j * np.zeros((Nfreq, NSOC))
for ii in range(len(SOCs)):
    modelhat.build(initial_state={"Initial SoC": SOCs[ii]})
    simulation = modelhat.simulateEIS(inputs=None, f_eval=frequencies)
    impedanceshat[:, ii] = simulation["Impedance"]

colors = plt.get_cmap("tab10")
fig, ax = plt.subplots()
for ii in range(len(SOCs)):
    ax.plot(np.real(impedances[:, ii]), -np.imag(impedances[:, ii]), color=colors(ii))
    ax.plot(
        np.real(impedanceshat[:, ii]), -np.imag(impedanceshat[:, ii]), color=colors(ii)
    )
ax.set(xlabel="$Z_r(\omega)$ [$\Omega$]", ylabel="$-Z_j(\omega)$ [$\Omega$]")
ax.grid()
ax.set_aspect("equal", "box")
plt.show()

mdic = {
    "Z": impedances,
    "Zhat": impedanceshat,
    "f": frequencies,
    "SOC": SOCs,
    "thetahat": thetahat,
    "computationTime": computationTime,
}
# savemat("Data/Zhat3mV_SOC_SPMe_LGM50LT.mat", mdic)
