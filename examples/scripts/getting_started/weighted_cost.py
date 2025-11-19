import numpy as np
import pybamm

import pybop

# Define model and parameter values
model = pybamm.lithium_ion.SPM()
parameter_values = pybamm.ParameterValues("Chen2020")
parameter_values.set_initial_state(0.5)

# Generate a synthetic dataset
sigma = 0.001
experiment = pybamm.Experiment(
    [
        "Discharge at 0.5C for 3 minutes (3 second period)",
        "Charge at 0.5C for 3 minutes (3 second period)",
    ]
    * 2
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
        "Negative electrode active material volume fraction",
        "Positive electrode active material volume fraction",
    ]
]

# Fitting parameters
parameter_values.update(
    {
        "Negative electrode active material volume fraction": pybop.ParameterDistribution(
            distribution=pybop.Gaussian(0.68, 0.05, truncated_at=[0.5, 0.8])
        ),
        "Positive electrode active material volume fraction": pybop.ParameterDistribution(
            distribution=pybop.Gaussian(
                0.58,
                0.05,
                truncated_at=[0.4, 0.7],
            )
        ),
    }
)

# Generate problem, cost function, and optimisation class
simulator = pybop.pybamm.Simulator(
    model, parameter_values=parameter_values, protocol=dataset
)
cost1 = pybop.SumSquaredError(dataset)
cost2 = pybop.RootMeanSquaredError(dataset)
weighted_cost = pybop.WeightedCost(cost1, cost2, weights=[0.1, 1])
options = pybop.PintsOptions(verbose=True, max_iterations=60)

for cost in [weighted_cost, cost1, cost2]:
    problem = pybop.Problem(simulator, cost)
    optim = pybop.IRPropMin(problem, options=options)

    # Run the optimisation
    result = optim.run()
    print("True parameters:", true_values)

    # Plot the timeseries output
    pybop.plot.problem(problem, problem_inputs=result.x, title="Optimised Comparison")

    # Plot the optimisation result
    result.plot_convergence()
    result.plot_surface()
