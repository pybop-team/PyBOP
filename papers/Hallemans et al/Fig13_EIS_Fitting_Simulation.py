from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.io import savemat

import pybop
from pybop.models.lithium_ion.basic_SPMe import BaseGroupedSPMe

# To duplicate paper results, modify the below:
n_runs = 1  # 10
max_iterations = 10  # 1000

## Define true parameters
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

## Construct model
model = pybop.lithium_ion.GroupedSPMe(
    parameter_set=grouped_parameters, eis=True, var_pts=var_pts, options=model_options
)

## Parameter bounds for optimisation
tau_d_bounds = [5e2, 1e4]
tau_e_bounds = [2e2, 1e3]
zeta_bounds = [0.5, 1.5]
Qe_bounds = [5e2, 1e3]
tau_ct_bounds = [1e3, 5e4]
C_bounds = [1e-5, 1]
c0p_bounds = [0.8, 0.9]
c0n_bounds = [1e-5, 0.1]
c100p_bounds = [0.2, 0.3]
c100n_bounds = [0.85, 0.95]
t_plus_bounds = [0.2, 0.5]
R0_bounds = [1e-5, 0.05]


# Create the parameters object
parameters = pybop.Parameters(
    pybop.Parameter(
        "Series resistance [Ohm]",
        bounds=R0_bounds,
        prior=pybop.Uniform(*R0_bounds),
        initial_value=np.mean(R0_bounds),
        true_value=grouped_parameters["Series resistance [Ohm]"],
        transformation=pybop.LogTransformation(),
    ),
    pybop.Parameter(
        "Positive particle diffusion time scale [s]",
        bounds=tau_d_bounds,
        initial_value=np.mean(tau_d_bounds),
        prior=pybop.Uniform(*tau_d_bounds),
        true_value=grouped_parameters["Positive particle diffusion time scale [s]"],
        transformation=pybop.LogTransformation(),
    ),
    pybop.Parameter(
        "Negative particle diffusion time scale [s]",
        bounds=tau_d_bounds,
        initial_value=np.mean(tau_d_bounds),
        prior=pybop.Uniform(*tau_d_bounds),
        true_value=grouped_parameters["Negative particle diffusion time scale [s]"],
        transformation=pybop.LogTransformation(),
    ),
    pybop.Parameter(
        "Cation transference number",
        bounds=t_plus_bounds,
        initial_value=np.mean(t_plus_bounds),
        prior=pybop.Uniform(*t_plus_bounds),
        true_value=grouped_parameters["Cation transference number"],
        transformation=pybop.LogTransformation(),
    ),
    pybop.Parameter(
        "Positive electrode electrolyte diffusion time scale [s]",
        bounds=tau_e_bounds,
        initial_value=np.mean(tau_e_bounds),
        prior=pybop.Uniform(*tau_e_bounds),
        true_value=grouped_parameters[
            "Positive electrode electrolyte diffusion time scale [s]"
        ],
        transformation=pybop.LogTransformation(),
    ),
    pybop.Parameter(
        "Negative electrode electrolyte diffusion time scale [s]",
        bounds=tau_e_bounds,
        initial_value=np.mean(tau_e_bounds),
        prior=pybop.Uniform(*tau_e_bounds),
        true_value=grouped_parameters[
            "Negative electrode electrolyte diffusion time scale [s]"
        ],
        transformation=pybop.LogTransformation(),
    ),
    pybop.Parameter(
        "Separator electrolyte diffusion time scale [s]",
        bounds=tau_e_bounds,
        initial_value=np.mean(tau_e_bounds),
        prior=pybop.Uniform(*tau_e_bounds),
        true_value=grouped_parameters["Separator electrolyte diffusion time scale [s]"],
        transformation=pybop.LogTransformation(),
    ),
    pybop.Parameter(
        "Positive electrode charge transfer time scale [s]",
        bounds=tau_ct_bounds,
        initial_value=np.mean(tau_ct_bounds),
        prior=pybop.Uniform(*tau_ct_bounds),
        true_value=grouped_parameters[
            "Positive electrode charge transfer time scale [s]"
        ],
        transformation=pybop.LogTransformation(),
    ),
    pybop.Parameter(
        "Negative electrode charge transfer time scale [s]",
        bounds=tau_ct_bounds,
        initial_value=np.mean(tau_ct_bounds),
        prior=pybop.Uniform(*tau_ct_bounds),
        true_value=grouped_parameters[
            "Negative electrode charge transfer time scale [s]"
        ],
        transformation=pybop.LogTransformation(),
    ),
    pybop.Parameter(
        "Positive electrode capacitance [F]",
        bounds=C_bounds,
        initial_value=np.mean(C_bounds),
        prior=pybop.Uniform(*C_bounds),
        true_value=grouped_parameters["Positive electrode capacitance [F]"],
        transformation=pybop.LogTransformation(),
    ),
    pybop.Parameter(
        "Negative electrode capacitance [F]",
        bounds=C_bounds,
        initial_value=np.mean(C_bounds),
        prior=pybop.Uniform(*C_bounds),
        true_value=grouped_parameters["Negative electrode capacitance [F]"],
        transformation=pybop.LogTransformation(),
    ),
    pybop.Parameter(
        "Positive electrode relative porosity",
        bounds=zeta_bounds,
        initial_value=np.mean(zeta_bounds),
        prior=pybop.Uniform(*zeta_bounds),
        true_value=grouped_parameters["Positive electrode relative porosity"],
        transformation=pybop.LogTransformation(),
    ),
    pybop.Parameter(
        "Negative electrode relative porosity",
        bounds=zeta_bounds,
        initial_value=np.mean(zeta_bounds),
        prior=pybop.Uniform(*zeta_bounds),
        true_value=grouped_parameters["Negative electrode relative porosity"],
        transformation=pybop.LogTransformation(),
    ),
    pybop.Parameter(
        "Reference electrolyte capacity [A.s]",
        bounds=Qe_bounds,
        initial_value=np.mean(Qe_bounds),
        prior=pybop.Uniform(*Qe_bounds),
        true_value=grouped_parameters["Reference electrolyte capacity [A.s]"],
        transformation=pybop.LogTransformation(),
    ),
    pybop.Parameter(
        "Minimum positive stoichiometry",
        bounds=c100p_bounds,
        initial_value=np.mean(c100p_bounds),
        prior=pybop.Uniform(*c100p_bounds),
        true_value=grouped_parameters["Minimum positive stoichiometry"],
        transformation=pybop.LogTransformation(),
    ),
    pybop.Parameter(
        "Maximum positive stoichiometry",
        bounds=c0p_bounds,
        initial_value=np.mean(c0p_bounds),
        prior=pybop.Uniform(*c0p_bounds),
        true_value=grouped_parameters["Maximum positive stoichiometry"],
        transformation=pybop.LogTransformation(),
    ),
    pybop.Parameter(
        "Minimum negative stoichiometry",
        bounds=c0n_bounds,
        initial_value=np.mean(c0n_bounds),
        prior=pybop.Uniform(*c0n_bounds),
        true_value=grouped_parameters["Minimum negative stoichiometry"],
        transformation=pybop.LogTransformation(),
    ),
    pybop.Parameter(
        "Maximum negative stoichiometry",
        bounds=c100n_bounds,
        initial_value=np.mean(c100n_bounds),
        prior=pybop.Uniform(*c100n_bounds),
        true_value=grouped_parameters["Maximum negative stoichiometry"],
        transformation=pybop.LogTransformation(),
    ),
)


## Read simulated impedance data
current_dir = Path(__file__).parent
EIS_data_path = current_dir / "Data" / "Z_SPMegrouped_SOC_chen2020.mat"
EIS_data = scipy.io.loadmat(EIS_data_path)

impedances = EIS_data.get("Z")
frequencies = EIS_data.get("f")
frequencies = frequencies.flatten()
SOCs = EIS_data.get("SOC")
SOCs = SOCs.flatten()

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

problem = pybop.MultiFittingProblem(*problems)
cost = pybop.SumSquaredError(problem)
optim = pybop.PSO(
    cost,
    parallel=True,
    multistart=n_runs,
    max_iterations=max_iterations,
    max_unchanged_iterations=max_iterations,
    # polish=False, # For SciPyDifferential
    # popsize=5, # For SciPyDifferential
)

results = optim.run()

# Print optimised parameters
print("True grouped parameters", parameters.true_value())
grouped_parameters.update(parameters.as_dict(results.best_x))

# Plot convergence
pybop.plot.convergence(optim)

# Plot the parameter traces
pybop.plot.parameters(optim)

## Save estimated impedance data
model_hat = pybop.lithium_ion.GroupedSPMe(
    parameter_set=grouped_parameters,
    eis=True,
    var_pts=var_pts,
    options=model_options,
)

Nfreq = len(frequencies)
NSOC = len(SOCs)

# Compute impedance for estimated parameters
impedances_hat = 1j * np.zeros((Nfreq, NSOC))
for ii in range(len(SOCs)):
    model_hat.build(initial_state={"Initial SoC": SOCs[ii]})
    simulation = model_hat.simulateEIS(inputs=None, f_eval=frequencies)
    impedances_hat[:, ii] = simulation["Impedance"]

colors = plt.get_cmap("tab10")
fig, ax = plt.subplots()
for ii in range(len(SOCs)):
    ax.plot(np.real(impedances[:, ii]), -np.imag(impedances[:, ii]), color=colors(ii))
    ax.plot(
        np.real(impedances_hat[:, ii]),
        -np.imag(impedances_hat[:, ii]),
        color=colors(ii),
    )
ax.set(xlabel=r"$Z_r(\omega)$ [$\Omega$]", ylabel=r"$-Z_j(\omega)$ [$\Omega$]")
ax.grid()
ax.set_aspect("equal", "box")
plt.show()

mdic = {
    "Z": impedances,
    "Zhat": impedances_hat,
    "f": frequencies,
    "SOC": SOCs,
    "final_cost": results.final_cost,
    "theta": parameters.true_value(),
    "thetahat": results.x,
    "thetahatbest": results.best_x,
    "computationTime": results.time,
}
save_path = current_dir / "Data" / "Zhat_SOC_SPMe_Simulation.mat"
savemat(save_path, mdic)
