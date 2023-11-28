import pybop
import numpy as np
import matplotlib.pyplot as plt

# parameter_set = pybop.ParameterSet("pybamm", "Chen2020")
model = pybop.empirical.Thevenin()

# Fitting parameters
parameters = [
    pybop.Parameter(
        "R0 [Ohm]",
        prior=pybop.Gaussian(0.001, 0.0001),
        bounds=[1e-5, 1e-2],
    ),
    pybop.Parameter(
        "R1 [Ohm]",
        prior=pybop.Gaussian(0.001, 0.0001),
        bounds=[1e-5, 1e-2],
    ),
]

sigma = 0.001
t_eval = np.arange(0, 900, 2)
values = model.predict(t_eval=t_eval)
CorruptValues = values["Battery voltage [V]"].data + np.random.normal(
    0, sigma, len(t_eval)
)

dataset = [
    pybop.Dataset("Time [s]", t_eval),
    pybop.Dataset("Current function [A]", values["Current [A]"].data),
    pybop.Dataset("Battery voltage [V]", CorruptValues),
]

# Generate problem, cost function, and optimisation class
problem = pybop.Problem(model, parameters, dataset)
cost = pybop.SumSquaredError(problem)
optim = pybop.Optimisation(cost, optimiser=pybop.CMAES)
optim.set_max_iterations(100)

x, final_cost = optim.run()
print("Estimated parameters:", x)

# Show the generated data
simulated_values = problem.evaluate(x)

plt.figure(dpi=100)
plt.xlabel("Time", fontsize=12)
plt.ylabel("Values", fontsize=12)
plt.plot(t_eval, CorruptValues, label="Measured")
plt.fill_between(t_eval, simulated_values - sigma, simulated_values + sigma, alpha=0.2)
plt.plot(t_eval, simulated_values, label="Simulated")
plt.legend(bbox_to_anchor=(0.6, 1), loc="upper left", fontsize=12)
plt.tick_params(axis="both", labelsize=12)
plt.show()
