import numpy as np

import pybop

# Parameter set and model definition
parameter_set = pybop.ParameterSet("Chen2020")
model = pybop.lithium_ion.SPM(parameter_set=parameter_set)

# Fitting parameters
parameters = pybop.Parameters(
    pybop.Parameter(
        "Negative electrode active material volume fraction",
        prior=pybop.Gaussian(0.68, 0.05),
        bounds=[0.5, 0.8],
        true_value=parameter_set["Negative electrode active material volume fraction"],
    ),
    pybop.Parameter(
        "Positive electrode active material volume fraction",
        prior=pybop.Gaussian(0.58, 0.05),
        bounds=[0.4, 0.7],
        true_value=parameter_set["Positive electrode active material volume fraction"],
    ),
)

# Generate data
sigma = 0.001
experiment = pybop.Experiment(
    [
        (
            "Discharge at 0.5C for 3 minutes (3 second period)",
            "Charge at 0.5C for 3 minutes (3 second period)",
        ),
    ]
    * 2
)
values = model.predict(experiment=experiment, initial_state={"Initial SoC": 0.5})


def noisy(data, sigma):
    return data + np.random.normal(0, sigma, len(data))


# Form dataset
dataset = pybop.Dataset(
    {
        "Time [s]": values["Time [s]"].data,
        "Current function [A]": values["Current [A]"].data,
        "Voltage [V]": noisy(values["Voltage [V]"].data, sigma),
    }
)

# Generate problem, cost function, and optimisation class
problem = pybop.FittingProblem(model, parameters, dataset)
cost1 = pybop.SumSquaredError(problem)
cost2 = pybop.RootMeanSquaredError(problem)
weighted_cost = pybop.WeightedCost(cost1, cost2, weights=[0.1, 1])

for cost in [weighted_cost, cost1, cost2]:
    optim = pybop.IRPropMin(cost, max_iterations=60)

    # Run the optimisation
    results = optim.run()
    print("True parameters:", parameters.true_value())

    # Plot the timeseries output
    pybop.plot.problem(problem, problem_inputs=results.x, title="Optimised Comparison")

    # Plot convergence
    pybop.plot.convergence(optim)

    # Plot the cost landscape with optimisation path
    pybop.plot.surface(optim)
