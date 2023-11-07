import pybop
import numpy as np
import matplotlib.pyplot as plt

model = pybop.lithium_ion.SPMe()
model.signal = "Terminal voltage [V]"

# Fitting parameters
parameters = [
    pybop.Parameter(
        "Negative electrode active material volume fraction",
        prior=pybop.Gaussian(0.53, 0.02),
        bounds=[0.6, 0.9],
    ),
    pybop.Parameter(
        "Positive electrode active material volume fraction",
        prior=pybop.Gaussian(0.62, 0.02),
        bounds=[0.5, 0.8],
    ),
    # pybop.Parameter(
    #     "Current function [A]",
    #     prior=pybop.Gaussian(1.1, 0.05),
    #     bounds=[0.8, 1.3],
    # ),
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
cost = pybop.RootMeanSquaredError(problem)

x0 = np.array([0.53, 0.62, 1.1])
opt = pybop.Optimisation(cost, optimiser=pybop.GradientDescent())

opt.optimiser.set_learning_rate = 0.025
opt.optimiser.set_max_unchanged_iterations=50
opt.optimiser.set_max_iterations=200

x1, f1 = opt.run()
print("Estimated parameters:")
print(x1)

# Show the generated data
simulated_values = problem.evaluate(x1[:3])

plt.figure()
plt.xlabel("Time")
plt.ylabel("Values")
plt.plot(time, CorruptValues)
plt.fill_between(time, simulated_values - sigma, simulated_values + sigma, alpha=0.2)
plt.plot(time, simulated_values)
plt.show()
