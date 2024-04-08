import numpy as np
import pybamm

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
problem = pybop.GITT(
    model="Weppner & Huggins", parameter_set=parameter_set, dataset=dataset
)
cost = pybop.RootMeanSquaredError(problem)

# Build the optimisation problem
optim = pybop.Optimisation(cost=cost, optimiser=pybop.PSO, verbose=True)

# Run the optimisation problem
x, final_cost = optim.run()
print("Estimated parameters:", x)

# Plot the timeseries output
pybop.quick_plot(problem, parameter_values=x, title="Optimised Comparison")

# Plot convergence
pybop.plot_convergence(optim)

# Plot the parameter traces
pybop.plot_parameters(optim)
