import numpy as np
import pybamm

import pybop

# Define model and parameter values
model = pybop.ExponentialDecayModel(n_states=2)
parameter_values = pybamm.ParameterValues({"k": 1, "y0": 0.5})

# Generate a synthetic dataset
sigma = 0.003
t_eval = np.linspace(0, 10, 100)
sol = pybamm.Simulation(model, parameter_values=parameter_values).solve(t_eval=t_eval)


def noisy(data, sigma):
    return data + np.random.normal(0, sigma, len(data))


dataset = pybop.Dataset(
    {
        "Time [s]": t_eval,
        "y_0": noisy(sol["y_0"](t_eval), sigma),
        "y_1": noisy(sol["y_1"](t_eval), sigma),
    }
)

# Fitting parameters
parameter_values.update(
    {
        "k": pybop.Gaussian(0.5, 0.05),
        "y0": pybop.Gaussian(0.2, 0.05),
    }
)

# Build the problem
simulator = pybop.pybamm.Simulator(
    model, parameter_values=parameter_values, protocol=dataset
)
cost = pybop.Minkowski(dataset, target=["y_0", "y_1"], p=2)
problem = pybop.Problem(simulator, cost)

# Set up the optimiser
options = pybop.PintsOptions(
    verbose=True,
    sigma=0.02,
    max_iterations=100,
    max_unchanged_iterations=20,
)
optim = pybop.AdamW(problem, options=options)

# Run the optimisation
result = optim.run()

# Plot the timeseries output
pybop.plot.problem(problem, problem_inputs=result.x, title="Optimised Comparison")

# Plot the optimisation result
result.plot_convergence()
result.plot_parameters()
result.plot_surface(bounds=[[0, 2], [0, 1]])
