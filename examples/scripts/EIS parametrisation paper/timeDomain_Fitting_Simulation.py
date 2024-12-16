import time as timer

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.io import savemat

import pybop
from pybop.models.lithium_ion.basic_SPMe import convert_physical_to_grouped_parameters

Nruns = 10

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
    parameter_set=grouped_parameters, var_pts=var_pts, options=model_options
)

## Choose parameter bounds for optimisation
R0_bounds = [1e-5, 0.05]
tau_d_bounds = [5e2, 1e4]
t_plus_bounds = [0.2, 0.5]
tau_e_bounds = [2e2, 1e3]
tau_ct_bounds = [1e3, 5e4]
C_bounds = [1e-5, 1]
zeta_bounds = [0.5, 1.5]
Qe_bounds = [5e2, 1e3]
# c0p_bounds = [0.8, 0.9]
# c0n_bounds = [1e-5, 0.1]
# c100p_bounds = [0.2, 0.3]
# c100n_bounds = [0.85, 0.95]


parameters = pybop.Parameters(
    pybop.Parameter(
        "Series resistance [Ohm]",
        bounds=R0_bounds,
        true_value=grouped_parameters["Series resistance [Ohm]"],
        # prior = pybop.Gaussian(np.mean(R0_bounds), 1e-5),
        initial_value=np.mean(R0_bounds),
    ),
    pybop.Parameter(
        "Positive particle diffusion time scale [s]",
        bounds=tau_d_bounds,
        true_value=grouped_parameters["Positive particle diffusion time scale [s]"],
        # prior = pybop.Gaussian(np.mean(tau_d_bounds), 1e-5),
        initial_value=np.mean(tau_d_bounds),
        transformation=pybop.LogTransformation(),
    ),
    pybop.Parameter(
        "Negative particle diffusion time scale [s]",
        bounds=tau_d_bounds,
        true_value=grouped_parameters["Negative particle diffusion time scale [s]"],
        # prior=pybop.Gaussian(np.mean(tau_d_bounds), 1e-5),
        initial_value=np.mean(tau_d_bounds),
        transformation=pybop.LogTransformation(),
    ),
    pybop.Parameter(
        "Cation transference number",
        bounds=t_plus_bounds,
        true_value=grouped_parameters["Cation transference number"],
        # prior=pybop.Gaussian(np.mean(t_plus_bounds), 1e-5),
        initial_value=np.mean(t_plus_bounds),
        transformation=pybop.LogTransformation(),
    ),
    pybop.Parameter(
        "Positive electrode electrolyte diffusion time scale [s]",
        bounds=tau_e_bounds,
        true_value=grouped_parameters[
            "Positive electrode electrolyte diffusion time scale [s]"
        ],
        # prior=pybop.Gaussian(np.mean(tau_e_bounds), 1e-5),
        initial_value=np.mean(tau_e_bounds),
        transformation=pybop.LogTransformation(),
    ),
    pybop.Parameter(
        "Negative electrode electrolyte diffusion time scale [s]",
        bounds=tau_e_bounds,
        true_value=grouped_parameters[
            "Negative electrode electrolyte diffusion time scale [s]"
        ],
        # prior=pybop.Gaussian(np.mean(tau_e_bounds), 1e-5),
        initial_value=np.mean(tau_e_bounds),
        transformation=pybop.LogTransformation(),
    ),
    pybop.Parameter(
        "Separator electrolyte diffusion time scale [s]",
        bounds=tau_e_bounds,
        true_value=grouped_parameters["Separator electrolyte diffusion time scale [s]"],
        # prior=pybop.Gaussian(np.mean(tau_e_bounds), 1e-5),
        initial_value=np.mean(tau_e_bounds),
        transformation=pybop.LogTransformation(),
    ),
    pybop.Parameter(
        "Positive electrode charge transfer time scale [s]",
        bounds=tau_ct_bounds,
        true_value=grouped_parameters[
            "Positive electrode charge transfer time scale [s]"
        ],
        # prior=pybop.Gaussian(np.mean(tau_ct_bounds), 1e-5),
        initial_value=np.mean(tau_ct_bounds),
        transformation=pybop.LogTransformation(),
    ),
    pybop.Parameter(
        "Negative electrode charge transfer time scale [s]",
        bounds=tau_ct_bounds,
        true_value=grouped_parameters[
            "Negative electrode charge transfer time scale [s]"
        ],
        # prior=pybop.Gaussian(np.mean(tau_ct_bounds), 1e-5),
        initial_value=np.mean(tau_ct_bounds),
        transformation=pybop.LogTransformation(),
    ),
    pybop.Parameter(
        "Positive electrode capacitance [F]",
        bounds=C_bounds,
        true_value=grouped_parameters["Positive electrode capacitance [F]"],
        # prior=pybop.Gaussian(np.mean(C_bounds), 1e-5),
        initial_value=np.mean(C_bounds),
        transformation=pybop.LogTransformation(),
    ),
    pybop.Parameter(
        "Negative electrode capacitance [F]",
        bounds=C_bounds,
        true_value=grouped_parameters["Negative electrode capacitance [F]"],
        # prior=pybop.Gaussian(np.mean(C_bounds), 1e-5),
        initial_value=np.mean(C_bounds),
        transformation=pybop.LogTransformation(),
    ),
    pybop.Parameter(
        "Positive electrode relative porosity",
        bounds=zeta_bounds,
        true_value=grouped_parameters["Positive electrode relative porosity"],
        # prior=pybop.Gaussian(np.mean(zeta_bounds), 1e-5),
        initial_value=np.mean(zeta_bounds),
        transformation=pybop.LogTransformation(),
    ),
    pybop.Parameter(
        "Negative electrode relative porosity",
        bounds=zeta_bounds,
        true_value=grouped_parameters["Negative electrode relative porosity"],
        # prior=pybop.Gaussian(np.mean(zeta_bounds), 1e-5),
        initial_value=np.mean(zeta_bounds),
        transformation=pybop.LogTransformation(),
    ),
    pybop.Parameter(
        "Reference electrolyte capacity [A.s]",
        bounds=Qe_bounds,
        true_value=grouped_parameters["Reference electrolyte capacity [A.s]"],
        # prior=pybop.Gaussian(np.mean(Qe_bounds), 1e-5),
        initial_value=np.mean(Qe_bounds),
        transformation=pybop.LogTransformation(),
    ),
    # pybop.Parameter(
    #     "Minimum positive stoichiometry",
    #     true_value=grouped_parameters["Minimum positive stoichiometry"],
    #     bounds=c100p_bounds,
    #     initial_value=np.mean(c100p_bounds),
    #     transformation=pybop.LogTransformation()
    # ),
    # pybop.Parameter(
    #     "Maximum positive stoichiometry",
    #     bounds=c0p_bounds,
    #     true_value=grouped_parameters["Maximum positive stoichiometry"],
    #     initial_value=np.mean(c0p_bounds),
    #     transformation=pybop.LogTransformation()
    # ),
    # pybop.Parameter(
    #     "Minimum negative stoichiometry",
    #     true_value=grouped_parameters["Minimum negative stoichiometry"],
    #     bounds=c0n_bounds,
    #     initial_value=np.mean(c0n_bounds),
    #     transformation=pybop.LogTransformation()
    # ),
    # pybop.Parameter(
    #     "Maximum negative stoichiometry",
    #     true_value=grouped_parameters["Maximum negative stoichiometry"],
    #     bounds=c100n_bounds,
    #     initial_value=np.mean(c100n_bounds),
    #     transformation=pybop.LogTransformation()
    # ),
)

## Read simulated time domain data
timeDomainData = scipy.io.loadmat("Data/timeDomainSimulation_SPMegrouped.mat")
SOC0 = timeDomainData.get("SOC0")
t = timeDomainData.get("t")
i = timeDomainData.get("i")
v = timeDomainData.get("v")

SOC0 = SOC0.flatten()
SOC0 = SOC0[0]

t = t.flatten()
i = i.flatten()
v = v.flatten()

model.build(initial_state={"Initial SoC": SOC0})
dataset = pybop.Dataset(
    {
        "Time [s]": t,
        "Current function [A]": i,
        "Voltage [V]": v,
    }
)
problem = pybop.FittingProblem(model, parameters, dataset)
cost = pybop.SumSquaredError(problem)
# optim = pybop.SciPyDifferentialEvolution(
#     cost,
#     maxiter=100,
# )

optim = pybop.PSO(
    cost,
    parallel=True,
    max_iterations=1000,
    max_unchanged_iterations=1000,
    # sigma0=0.1
)

# Run optimisation
computationTime = np.zeros(Nruns)
final_cost = np.zeros(Nruns)
thetahat = np.zeros((len(parameters), Nruns))
# Run optimisation
for ii in range(Nruns):
    start_time = timer.time()
    results = optim.run()
    thetahat[:, ii] = results.x
    end_time = timer.time()
    computationTime[ii] = end_time - start_time
    final_cost[ii] = results.final_cost
    print(ii)

thetahatmean = np.mean(thetahat, axis=1)

# Print optimised parameters
print("True grouped parameters", parameters.true_value())

optimised_grouped_parameters = grouped_parameters
optimised_grouped_parameters.update(parameters.as_dict(thetahatmean))

grouped_parameters["Lower voltage cut-off [V]"] = 2.5
grouped_parameters["Upper voltage cut-off [V]"] = 4.2

# print("Optimised grouped parameters:", optimised_grouped_parameters)

# Plot convergence
pybop.plot.convergence(optim)

# Plot the parameter traces
pybop.plot.parameters(optim)


## Save estimated voltage
modelhat = pybop.lithium_ion.GroupedSPMe(
    parameter_set=optimised_grouped_parameters, var_pts=var_pts, options=model_options
)

modelhat.build(initial_state={"Initial SoC": SOC0})

# Grouped SPMe
modelhat.set_current_function(dataset=dataset)
simulationhat = modelhat.predict(t_eval=dataset["Time [s]"])
datasethat = pybop.Dataset(
    {
        "Time [s]": simulationhat["Time [s]"].data,
        "Current function [A]": simulationhat["Current [A]"].data,
        "Voltage [V]": simulationhat["Voltage [V]"].data,
    }
)

# Compare
fig, ax = plt.subplots()
ax.plot(dataset["Time [s]"], dataset["Voltage [V]"])
ax.plot(datasethat["Time [s]"], datasethat["Voltage [V]"])
ax.set(xlabel="time [s]", ylabel="Voltage [V]")
ax.grid()
plt.show()

## Save data
t = datasethat["Time [s]"].data
i = datasethat["Current function [A]"].data
vhat = datasethat["Voltage [V]"].data

mdic = {
    "t": t,
    "i": i,
    "vhat": vhat,
    "SOC0": SOC0,
    "final_cost": final_cost,
    "thetahat": thetahat,
    "computationTime": computationTime,
}
savemat("Data/Estimate_timeDomainSimulation.mat", mdic)
