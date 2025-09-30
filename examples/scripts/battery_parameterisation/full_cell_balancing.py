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

# Define fitting parameters for OCP balancing
parameters = pybop.Parameters(
    pybop.Parameter(
        "Maximum concentration in negative electrode [mol.m-3]",
        prior=pybop.Gaussian(cs_n_max, 6e3),
        bounds=[cs_n_max * 0.75, cs_n_max * 1.25],
        initial_value=cs_n_max * 0.8,
    ),
    pybop.Parameter(
        "Maximum concentration in positive electrode [mol.m-3]",
        prior=pybop.Gaussian(cs_p_max, 6e3),
        bounds=[cs_p_max * 0.75, cs_p_max * 1.25],
        initial_value=cs_p_max * 0.8,
    ),
    pybop.Parameter(
        "Initial concentration in negative electrode [mol.m-3]",
        prior=pybop.Gaussian(cs_n_init, 6e3),
        bounds=[cs_n_max * 0.75, cs_n_max * 1.25],
        initial_value=cs_n_max * 0.8,
    ),
    pybop.Parameter(
        "Initial concentration in positive electrode [mol.m-3]",
        prior=pybop.Gaussian(cs_p_init, 6e3),
        bounds=[0, cs_p_max * 0.5],
        initial_value=cs_p_max * 0.2,
    ),
)

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

# Build the problem
simulator = pybop.pybamm.Simulator(
    model, parameter_values, parameters=parameters, protocol=dataset
)
cost = pybop.GaussianLogLikelihoodKnownSigma(dataset, sigma0=sigma)
problem = pybop.Problem(simulator, cost)

# Set up the optimiser
options = pybop.SciPyMinimizeOptions(maxiter=125)
optim = pybop.SciPyMinimize(problem, options=options)

# Run the optimisation for Maximum Likelihood Estimate (MLE)
result = optim.run()
print(result)
print("True values:", [parameter_values[p] for p in parameters.keys()])

# Plot the timeseries output
pybop.plot.problem(problem, problem_inputs=result.x, title="Optimised Comparison")

# Plot the optimisation result
result.plot_convergence()
result.plot_parameters()
result.plot_contour(steps=5)
