import numpy as np
import pybamm

import pybop

"""
In this example, we introduce the Adaptive Moment Estimation with Weight Decay (AdamW)
optimisation algorithm. This optimiser uses gradient information for trajectory and
step-size determination.
"""

# Define model and parameter values
model = pybamm.lithium_ion.SPM()
parameter_values = pybamm.ParameterValues("Chen2020")

# Generate a synthetic dataset
sigma = 5e-3
t_eval = np.linspace(0, 500, 240)
solution = pybamm.Simulation(model, parameter_values=parameter_values).solve(
    t_eval=t_eval
)
dataset = pybop.Dataset(
    {
        "Time [s]": t_eval,
        "Voltage [V]": solution["Voltage [V]"](t_eval)
        + np.random.normal(0, sigma, len(t_eval)),
        "Current function [A]": solution["Current [A]"](t_eval),
    }
)

# Fitting parameters
parameter_values.update(
    {
        "Negative electrode active material volume fraction": pybop.Parameter(
            prior=pybop.Gaussian(0.68, 0.05),
            initial_value=0.45,
            bounds=[0.4, 0.9],
        ),
        "Positive electrode active material volume fraction": pybop.Parameter(
            prior=pybop.Gaussian(0.58, 0.05),
            initial_value=0.45,
            bounds=[0.4, 0.9],
        ),
    }
)

# Build the problem
simulator = pybop.pybamm.Simulator(
    model, parameter_values=parameter_values, protocol=dataset
)
cost = pybop.SumOfPower(dataset, p=2.5)
problem = pybop.Problem(simulator, cost)

# Set up the optimiser
options = pybop.PintsOptions(
    verbose=True,
    verbose_print_rate=20,
    max_iterations=150,
    max_unchanged_iterations=40,
)
optim = pybop.AdamW(problem, options=options)

# Reduce the momentum influence for the reduced number of optimiser iterations
optim.optimiser.b1 = 0.75
optim.optimiser.b2 = 0.75

# Run the optimisation
result = optim.run()

# Plot the timeseries output
pybop.plot.problem(problem, inputs=result.best_inputs, title="Optimised Comparison")

# Plot the optimisation result
result.plot_convergence()
result.plot_parameters()
result.plot_surface()
