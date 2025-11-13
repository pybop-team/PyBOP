import numpy as np
import pybamm

import pybop

# Define model and parameter values
model = pybamm.lithium_ion.SPM()
parameter_values = pybamm.ParameterValues("Chen2020")
parameter_values["Lower voltage cut-off [V]"] = 2.3
parameter_values["Upper voltage cut-off [V]"] = 4.4

# Set initial state and unpack true values
parameter_values.set_initial_state(1.0)
cs_n_max = parameter_values["Maximum concentration in negative electrode [mol.m-3]"]
cs_p_max = parameter_values["Maximum concentration in positive electrode [mol.m-3]"]
cs_n_init = parameter_values["Initial concentration in negative electrode [mol.m-3]"]
cs_p_init = parameter_values["Initial concentration in positive electrode [mol.m-3]"]

# Generate a synthetic data
sigma = 5e-4
experiment = pybamm.Experiment(
    [
        "Discharge at 0.1C until 2.5V (3 min period)",
        "Charge at 0.1C until 4.2V (3 min period)",
    ]
)
sol = pybamm.Simulation(
    model, parameter_values=parameter_values, experiment=experiment
).solve()


def noisy(data, sigma):
    return data + np.random.normal(0, sigma, len(data))


dataset = pybop.Dataset(
    {
        "Time [s]": sol.t,
        "Current function [A]": sol["Current [A]"].data,
        "Voltage [V]": noisy(sol["Voltage [V]"].data, sigma),
    }
)
true_values = [
    parameter_values[p]
    for p in [
        "Maximum concentration in negative electrode [mol.m-3]",
        "Maximum concentration in positive electrode [mol.m-3]",
        "Initial concentration in negative electrode [mol.m-3]",
        "Initial concentration in positive electrode [mol.m-3]",
    ]
]

# Define fitting parameters for OCP balancing
parameter_values.update(
    {
        "Maximum concentration in negative electrode [mol.m-3]": pybop.Parameter(
            prior=pybop.Gaussian(cs_n_max, 6e3),
            bounds=[cs_n_max * 0.75, cs_n_max * 1.25],
            initial_value=cs_n_max * 0.8,
        ),
        "Maximum concentration in positive electrode [mol.m-3]": pybop.Parameter(
            prior=pybop.Gaussian(cs_p_max, 6e3),
            bounds=[cs_p_max * 0.75, cs_p_max * 1.25],
            initial_value=cs_p_max * 0.8,
        ),
        "Initial concentration in negative electrode [mol.m-3]": pybop.Parameter(
            prior=pybop.Gaussian(cs_n_init, 6e3),
            bounds=[cs_n_max * 0.75, cs_n_max * 1.25],
            initial_value=cs_n_max * 0.8,
        ),
        "Initial concentration in positive electrode [mol.m-3]": pybop.Parameter(
            prior=pybop.Gaussian(cs_p_init, 6e3),
            bounds=[0, cs_p_max * 0.5],
            initial_value=cs_p_max * 0.2,
        ),
    }
)

# Build the problem
simulator = pybop.pybamm.Simulator(model, parameter_values, protocol=dataset)
cost = pybop.GaussianLogLikelihoodKnownSigma(dataset, sigma0=sigma)
problem = pybop.Problem(simulator, cost)

# Set up the optimiser
options = pybop.SciPyMinimizeOptions(maxiter=125, method="Nelder-Mead")
optim = pybop.SciPyMinimize(problem, options=options)

# Run the optimisation for Maximum Likelihood Estimate (MLE)
result = optim.run()
print(result)
print("True values:", true_values)

# Plot the timeseries output
pybop.plot.problem(problem, inputs=result.best_inputs, title="Optimised Comparison")

# Plot the optimisation result
result.plot_convergence()
result.plot_parameters()
result.plot_contour(steps=5)
