import pybop
import pints
import numpy as np
import matplotlib.pyplot as plt

model = pybop.lithium_ion.SPMe()
model.parameter_set["Current function [A]"] = 1

inputs = {
    "Negative electrode active material volume fraction": 0.587,
    "Positive electrode active material volume fraction": 0.44,
}
t_eval = np.arange(0, 900, 2)
model.build(fit_parameters=inputs)

values = model.predict(inputs=inputs, t_eval=t_eval)
voltage = values["Terminal voltage [V]"].data
time = values["Time [s]"].data

sigma = 0.0
CorruptValues = voltage + np.random.normal(0, sigma, len(voltage))

# Show the generated data
plt.figure()
plt.xlabel("Time")
plt.ylabel("Values")
plt.plot(time, CorruptValues)
plt.plot(time, voltage)
plt.show()


problem = pints.SingleOutputProblem(model, time, CorruptValues)

# Select a score function
score = pints.SumOfSquaresError(problem)

x0 = np.array([0.5, 0.5])
opt = pints.OptimisationController(score, x0, method=pints.GradientDescent)

opt.optimiser().set_learning_rate(0.025)
opt.set_max_unchanged_iterations(250)
opt.set_max_iterations(500)


x1, f1 = opt.run()
print("Estimated parameters:")
print(x1)

# Show the generated data
simulated_values = problem.evaluate(x1[:2])

plt.figure()
plt.xlabel("Time")
plt.ylabel("Values")
plt.plot(time, CorruptValues)
plt.fill_between(time, simulated_values - sigma, simulated_values + sigma, alpha=0.2)
plt.plot(time, simulated_values)
plt.show()
