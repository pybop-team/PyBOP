import pybop
import numpy as np
import matplotlib.pyplot as plt

parameter_set = pybop.ParameterSet("pybamm", "Chen2020")
model = pybop.lithium_ion.SPM(parameter_set=parameter_set)

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

sigma = 0.001
t_eval = np.arange(0, 900, 2)
values = model.predict(t_eval=t_eval)
CorruptValues = values["Voltage [V]"].data + np.random.normal(0, sigma, len(t_eval))

dataset = [
    pybop.Dataset("Time [s]", t_eval),
    pybop.Dataset("Current function [A]", values["Current [A]"].data),
    pybop.Dataset("Voltage [V]", CorruptValues),
]

# Generate problem, cost function, and optimisation class
problem = pybop.Problem(model, parameters, dataset)
cost = pybop.SumSquaredError(problem)
optim = pybop.Optimisation(cost, optimiser=pybop.PSO)
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
