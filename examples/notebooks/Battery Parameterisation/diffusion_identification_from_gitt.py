import numpy as np

import pybop

# Define model
parameter_set = pybop.ParameterSet.pybamm("Xu2019")
model = pybop.lithium_ion.SPM(
    parameter_set=parameter_set, options={"working electrode": "positive"}
)

# Generate data
sigma = 0.005
t_eval = np.arange(0, 150, 2)
values = model.predict(t_eval=t_eval)
corrupt_values = values["Voltage [V]"].data + np.random.normal(0, sigma, len(t_eval))

# Form dataset
dataset = pybop.Dataset(
    {
        "Time [s]": t_eval,
        "Current function [A]": values["Current [A]"].data,
        "Voltage [V]": corrupt_values,
    }
)

# Define parameter set
parameter_set.update(
    {
        "Reference OCP [V]": 4.1821,
        "Derivative of the OCP wrt stoichiometry [V]": -1.38636,
    },
    check_already_exists=False,
)

# Define the cost to optimise
model = pybop.lithium_ion.WeppnerHuggins(parameter_set=parameter_set)

parameters = pybop.Parameter(
    "Positive electrode diffusivity [m2.s-1]",
    prior=pybop.Gaussian(5e-14, 1e-13),
    bounds=[1e-16, 1e-11],
    true_value=parameter_set["Positive electrode diffusivity [m2.s-1]"],
)

problem = pybop.FittingProblem(
    model,
    parameters,
    dataset,
    signal=["Voltage [V]"],
)

cost = pybop.RootMeanSquaredError(problem)

# Build the optimisation problem
optim = pybop.PSO(cost=cost, verbose=True)

# Run the optimisation problem
x, final_cost = optim.run()
print("Estimated parameters:", x)

# Plot the timeseries output
pybop.quick_plot(problem, parameter_values=x, title="Optimised Comparison")

# Plot convergence
pybop.plot_convergence(optim)

# Plot the parameter traces
pybop.plot_parameters(optim)
