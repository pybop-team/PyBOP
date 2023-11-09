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
corrupt_values = voltage + np.random.normal(0, sigma, len(voltage))

# Show the generated data
plt.figure()
#plt.title("Synthetic data and corrupted signal")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.plot(time, corrupt_values, label="corrupted signal")
plt.plot(time, voltage, label="synthetic data")
plt.legend(loc="upper right")
plt.show()

# Generate problem, cost function, and optimisation class
problem = pints.SingleOutputProblem(model, time, corrupt_values)
log_likelihood = pints.GaussianLogLikelihood(problem)
boundaries = pints.RectangularBoundaries([0.4, 0.4, 0.7, 1e-5], [0.6, 0.6, 2.1, 1e-1])
x0 = np.array([0.48, 0.55, 1.4, 1e-3])
optim = pints.OptimisationController(
    log_likelihood, x0, boundaries=boundaries, method=pints.CMAES
)
optim.set_max_unchanged_iterations(50)
optim.set_max_iterations(200)

# Run optimisation
x1, f1 = optim.run()
print("Estimated parameters:")
print(x1)

# Generate data using the estimated parameters
simulated_values = problem.evaluate(x1[:3])

# Show the generated data
plt.figure()
#plt.title("Corrupted signal and estimation")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.plot(time, corrupt_values, label="corrupted signal")
plt.fill_between(time, simulated_values - sigma, simulated_values + sigma, alpha=0.2)
plt.plot(time, simulated_values, label="estimation")
plt.legend(loc="upper right")
plt.show()
