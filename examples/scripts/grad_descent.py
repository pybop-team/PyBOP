import pybop
import numpy as np
import matplotlib.pyplot as plt

parameter_set = pybop.ParameterSet("pybamm", "Chen2020")
model = pybop.lithium_ion.SPMe(parameter_set=parameter_set)
model.signal = "Terminal voltage [V]"

# Fitting parameters
parameters = [
    pybop.Parameter(
        "Negative electrode active material volume fraction",
        prior=pybop.Gaussian(0.57, 0.05),
        bounds=[0.6, 0.9],
    ),
    pybop.Parameter(
        "Positive electrode active material volume fraction",
        prior=pybop.Gaussian(0.58, 0.05),
        bounds=[0.5, 0.8],
    ),
    pybop.Parameter(
        "Current function [A]",
        prior=pybop.Gaussian(1.2, 0.05),
        bounds=[0.8, 1.3],
    ),
]

model.parameter_set.update(
    {
        "Negative electrode active material volume fraction": 0.53,
        "Positive electrode active material volume fraction": 0.62,
        "Current function [A]": 1.1,
    }
)

t_eval = np.arange(0, 900, 2)
values = model.predict(t_eval=t_eval)
voltage = values["Terminal voltage [V]"].data
time = values["Time [s]"].data

sigma = 0.001
CorruptValues = voltage + np.random.normal(0, sigma, len(voltage))

dataset = [
    pybop.Dataset("Time [s]", time),
    pybop.Dataset("Current function [A]", values["Current [A]"].data),
    pybop.Dataset("Terminal voltage [V]", CorruptValues),
]

# Show the generated data
plt.figure()
plt.xlabel("Time")
plt.ylabel("Values")
plt.plot(time, CorruptValues)
plt.plot(time, voltage)
plt.show()

signal = "Terminal voltage [V]"
problem = pybop.Problem(model, parameters, signal, dataset)

# Select a score function
cost = pybop.SumSquaredError(problem)
opt = pybop.Optimisation(cost, optimiser=pybop.GradientDescent())

opt.optimiser.learning_rate = 0.025
opt.optimiser.max_unchanged_iterations=1
opt.optimiser.max_iterations=50

x, output, final_cost, num_evals = opt.run()
print("Estimated parameters:", x)

# Show the generated data
simulated_values = problem.evaluate(x[:3])

plt.figure(figsize=(5, 5), dpi=100, facecolor="w", edgecolor="k", linewidth=2, frameon=False)
plt.xlabel("Time", fontsize=20)
plt.ylabel("Values", fontsize=20)
plt.plot(time, CorruptValues, label="Measured")
plt.fill_between(time, simulated_values - sigma, simulated_values + sigma, alpha=0.2)
plt.plot(time, simulated_values, label="Simulated")
plt.legend(
    bbox_to_anchor=(0.85, 1), loc="upper left", fontsize=20
)
plt.tick_params(axis='both', labelsize=20)
plt.show()
