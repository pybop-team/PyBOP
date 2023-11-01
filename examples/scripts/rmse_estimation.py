import pybop
import pandas as pd

# Form observations
Measurements = pd.read_csv("examples/scripts/Chen_example.csv", comment="#").to_numpy()
dataset = [
    pybop.Dataset("Time [s]", Measurements[:, 0]),
    pybop.Dataset("Current function [A]", Measurements[:, 1]),
    pybop.Dataset("Voltage [V]", Measurements[:, 2]),
]

# Define model
# parameter_set = pybop.ParameterSet("pybamm", "Chen2020")
model = pybop.models.lithium_ion.SPM()

# Fitting parameters
parameters = [
    pybop.Parameter(
        "Negative electrode active material volume fraction",
        prior=pybop.Gaussian(0.75, 0.05),
        bounds=[0.65, 0.85],
    ),
    pybop.Parameter(
        "Positive electrode active material volume fraction",
        prior=pybop.Gaussian(0.65, 0.05),
        bounds=[0.55, 0.75],
    ),
]

# Define the cost to optimise
cost = pybop.RMSE()
signal = "Voltage [V]"


# Build the optimisation problem
parameterisation = pybop.Optimisation(
    cost=cost,
    model=model,
    optimiser=pybop.NLoptOptimize(n_param=len(parameters)),
    parameters=parameters,
    dataset=dataset,
    signal=signal,
)

# Run the optimisation problem
x, output, final_cost, num_evals = parameterisation.run()


# get MAP estimate, starting at a random initial point in parameter space
# parameterisation.map(x0=[p.sample() for p in params])

# or sample from posterior
# parameterisation.sample(1000, n_chains=4, ....)

# or SOBER
# parameterisation.sober()


# Optimisation = pybop.optimisation(model, cost=cost, parameters=parameters, observation=observation)
