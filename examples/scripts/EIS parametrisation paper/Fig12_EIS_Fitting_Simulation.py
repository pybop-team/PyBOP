import time as timer

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.io import savemat

import pybop
from pybop.models.lithium_ion.basic_SPMe import convert_physical_to_grouped_parameters

Nruns = 10

## Define true parameters
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

## Model
model = pybop.lithium_ion.GroupedSPMe(
    parameter_set=grouped_parameters, eis=True, var_pts=var_pts, options=model_options
)

## Choose parameter bounds for optimisation
tau_d_bounds = [5e2, 1e4]
tau_e_bounds = [2e2, 1e3]
zeta_bounds = [0.5, 1.5]
Qe_bounds = [5e2, 1e3]
tau_ct_bounds = [1e3, 5e4]
C_bounds = [0, 1]
c0p_bounds = [0.8, 0.9]
c0n_bounds = [0, 0.1]
c100p_bounds = [0.2, 0.3]
c100n_bounds = [0.85, 0.95]
t_plus_bounds = [0.2, 0.5]
R0_bounds = [0, 0.05]

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


## Read simulated impedance data
EIS_data = scipy.io.loadmat("Data/Z_SPMegrouped_SOC_chen2020.mat")

impedances = EIS_data.get("Z")
frequencies = EIS_data.get("f")
frequencies = frequencies.flatten()
SOCs = EIS_data.get("SOC")
SOCs = SOCs.flatten()

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
    problems[7],
    problems[8],
)

cost = pybop.SumSquaredError(problem)
optim = pybop.SciPyDifferentialEvolution(
    cost, max_iterations=100, max_unchanged_iterations=100
)

computationTime = np.zeros(Nruns)
thetahat = np.zeros((len(parameters), Nruns))
# Run optimisation
for ii in range(Nruns):
    start_time = timer.time()
    estimate = optim.run()
    thetahat[:, ii] = estimate.x
    end_time = timer.time()
    computationTime[ii] = end_time - start_time
    print(ii)

thetahatmean = np.mean(thetahat, axis=1)

# Print optimised parameters
print("True grouped parameters", grouped_parameters)

optimised_grouped_parameters = grouped_parameters
optimised_grouped_parameters.update(parameters.as_dict(thetahatmean))

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
ax.set(xlabel=r"$Z_r(\omega)$ [$\Omega$]", ylabel=r"$-Z_j(\omega)$ [$\Omega$]")
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
savemat("Data/Zhat_SOC_SPMe_Simulation.mat", mdic)
