import pybop
import numpy as np

# Parameter set and model definition
parameter_set = pybop.ParameterSet("pybamm", "Chen2020")
model = pybop.lithium_ion.SPMe(parameter_set=parameter_set)

# Fitting parameters
parameters = [
    pybop.Parameter(
        "Negative electrode active material volume fraction",
        prior=pybop.Gaussian(0.7, 0.05),
        bounds=[0.6, 0.9],
    ),
    pybop.Parameter(
        "Positive electrode active material volume fraction",
        prior=pybop.Gaussian(0.58, 0.05),
        bounds=[0.5, 0.8],
    ),
]

# Generate data
sigma = 0.005
t_eval = np.arange(0, 900, 2)
values = model.predict(t_eval=t_eval)
CorruptValues = values["Terminal voltage [V]"].data + np.random.normal(
    0, sigma, len(t_eval)
)

# Dataset definition
dataset = [
    pybop.Dataset("Time [s]", t_eval),
    pybop.Dataset("Current function [A]", values["Current [A]"].data),
    pybop.Dataset("Terminal voltage [V]", CorruptValues),
]

# Generate problem, cost function, and optimisation class
problem = pybop.Problem(model, parameters, dataset)
cost = pybop.SumSquaredError(problem)
optim = pybop.Optimisation(cost, optimiser=pybop.GradientDescent())
optim.optimiser.learning_rate = 0.025
optim.optimiser.max_iterations = 100

# Run optimisation
x, output, final_cost, num_evals = optim.run()
print("Estimated parameters:", x)
