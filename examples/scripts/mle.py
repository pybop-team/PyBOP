import pybop
import pints
import numpy as np
import matplotlib.pyplot as plt

model = pybop.lithium_ion.SPMe()

inputs = {
    "Negative electrode active material volume fraction": 0.58,
    "Positive electrode active material volume fraction": 0.44,
    "Current function [A]": 1,
}
t_eval = np.arange(0, 900, 2)
model.build(fit_parameters=inputs)

values = model.predict(inputs=inputs, t_eval=t_eval)
voltage = values["Terminal voltage [V]"].data
time = values["Time [s]"].data

sigma = 0.001
CorruptValues = voltage + np.random.normal(0, sigma, len(voltage))

# Show the generated data
plt.figure()
plt.xlabel("Time")
plt.ylabel("Values")
plt.plot(time, CorruptValues)
plt.plot(time, voltage)
plt.show()


problem = pints.SingleOutputProblem(model, time, CorruptValues)
log_likelihood = pints.GaussianLogLikelihood(problem)
boundaries = pints.RectangularBoundaries([0.4, 0.4, 0.7, 1e-5], [0.6, 0.6, 2.1, 1e-1])

x0 = np.array([0.48, 0.55, 1.4, 1e-3])
opt = pints.OptimisationController(
    log_likelihood, x0, boundaries=boundaries, method=pints.CMAES
)
opt.set_max_unchanged_iterations(50)
opt.set_max_iterations(200)

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
