import pybop
import pints
import numpy as np
import matplotlib.pyplot as plt

# Model definition
model = pybop.lithium_ion.SPMe()

# Input current and fitting parameters
inputs = {
    "Negative electrode active material volume fraction": 0.58,
    "Positive electrode active material volume fraction": 0.44,
    "Current function [A]": 1,
}
t_eval = np.arange(0, 900, 2)
model.build(fit_parameters=inputs)

# Generate data
values = model.predict(inputs=inputs, t_eval=t_eval)
voltage = values["Terminal voltage [V]"].data
time = values["Time [s]"].data

# Add noise
sigma = 0.001
CorruptValues = voltage + np.random.normal(0, sigma, len(voltage))

# Show the generated data
plt.figure()
plt.xlabel("Time")
plt.ylabel("Values")
plt.plot(time, CorruptValues)
plt.plot(time, voltage)
plt.draw() # use draw instead of show so that computation continues

# Generate problem
problem = pints.SingleOutputProblem(model, time, CorruptValues)

# Select a score function
score = pints.SumOfSquaresError(problem)

# Set the initial parameter values and optimisation class
x0 = np.array([0.48, 0.55, 1.4])
optim = pints.OptimisationController(score, x0, method=pints.GradientDescent)
optim.optimiser().set_learning_rate(0.025)
optim.set_max_unchanged_iterations(50)
optim.set_max_iterations(200)

# Run optimisation
x1, f1 = optim.run()
print("Estimated parameters:")
print(x1)

# Generate data using the estimated parameters
simulated_values = problem.evaluate(x1[:3])

# Show the estimated data
plt.figure()
plt.xlabel("Time")
plt.ylabel("Values")
plt.plot(time, CorruptValues)
plt.fill_between(time, simulated_values - sigma, simulated_values + sigma, alpha=0.2)
plt.plot(time, simulated_values)
plt.show()
