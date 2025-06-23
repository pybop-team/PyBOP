import numpy as np
import scipy
from scipy.io import savemat

import pybop
from pybop.models.lithium_ion.grouped_spme import convert_physical_to_grouped_parameters

# To duplicate paper results, modify the below:
n_runs = 1  # 10
max_iterations = 10  # 1000

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

## Construct model
model = pybop.lithium_ion.GroupedSPMe(
    parameter_set=grouped_parameters, var_pts=var_pts, options=model_options
)

## Parameter bounds for optimisation
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
    # pybop.Parameter(
    #     "Minimum positive stoichiometry",
    #     bounds=c100p_bounds,
    #     initial_value=np.mean(c100p_bounds),
    #     prior=pybop.Uniform(*c100p_bounds),
    #     true_value=grouped_parameters["Minimum positive stoichiometry"],
    #     transformation=pybop.LogTransformation(),
    # ),
    # pybop.Parameter(
    #     "Maximum positive stoichiometry",
    #     bounds=c0p_bounds,
    #     initial_value=np.mean(c0p_bounds),
    #     prior=pybop.Uniform(*c0p_bounds),
    #     true_value=grouped_parameters["Maximum positive stoichiometry"],
    #     transformation=pybop.LogTransformation(),
    # ),
    # pybop.Parameter(
    #     "Minimum negative stoichiometry",
    #     bounds=c0n_bounds,
    #     initial_value=np.mean(c0n_bounds),
    #     prior=pybop.Uniform(*c0n_bounds),
    #     true_value=grouped_parameters["Minimum negative stoichiometry"],
    #     transformation=pybop.LogTransformation(),
    # ),
    # pybop.Parameter(
    #     "Maximum negative stoichiometry",
    #     bounds=c100n_bounds,
    #     initial_value=np.mean(c100n_bounds),
    #     prior=pybop.Uniform(*c100n_bounds),
    #     true_value=grouped_parameters["Maximum negative stoichiometry"],
    #     transformation=pybop.LogTransformation(),
    # ),
)

## Read simulated time domain data
timeDomainData = scipy.io.loadmat("Data/timeDomainSimulation_SPMegrouped.mat")
SOC0 = timeDomainData.get("SOC0")
t = timeDomainData.get("t").flatten()
i = timeDomainData.get("i").flatten()
v = timeDomainData.get("v").flatten()

SOC0 = SOC0.flatten()
SOC0 = SOC0[0]

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

# Plot traces
pybop.plot.quick(problem, results.best_x)

# Plot convergence
pybop.plot.convergence(optim)

# Plot the parameter traces
pybop.plot.parameters(optim)

## Save estimated voltage
model_hat = pybop.lithium_ion.GroupedSPMe(
    parameter_set=grouped_parameters, var_pts=var_pts, options=model_options
)

# Grouped SPMe
model_hat.build(initial_state={"Initial SoC": SOC0})
model_hat.set_current_function(dataset=dataset)
simulation_hat = model_hat.predict(t_eval=dataset["Time [s]"])

## Save data
mdic = {
    "t": simulation_hat["Time [s]"].data,
    "i": simulation_hat["Current [A]"].data,
    "v": v,
    "vhat": simulation_hat["Voltage [V]"].data,
    "SOC0": SOC0,
    "final_cost": results.final_cost,
    "theta": parameters.true_value(),
    "thetahat": results.x,
    "thetahatbest": results.best_x,
    "computationTime": results.time,
}
savemat("Data/Estimate_timeDomainSimulation.mat", mdic)
