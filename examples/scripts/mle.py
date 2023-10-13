import pybop
import pints
import numpy as np
import matplotlib.pyplot as plt

model = pybop.lithium_ion.SPM()
model.parameter_set["Current function [A]"] = 2

inputs = {
    "Negative electrode active material volume fraction": 0.5,
    "Positive electrode active material volume fraction": 0.5,
}
t_eval = np.arange(0, 900, 2)
model.build(fit_parameters=inputs)

values = model.predict(inputs=inputs, t_eval=t_eval)
V = values["Terminal voltage [V]"].data
T = values["Time [s]"].data

sigma = 0.01
CorruptValues = V + np.random.normal(0, sigma, len(V))
# Show the generated data
plt.figure()
plt.xlabel("Time")
plt.ylabel("Values")
plt.plot(T, CorruptValues)
plt.plot(T, V)
plt.show()


problem = pints.SingleOutputProblem(model, T, CorruptValues)
# cost = pints.SumOfSquaresError(problem)

log_likelihood = pints.GaussianLogLikelihood(problem)
boundaries = pints.RectangularBoundaries([0.4, 0.4, 1e-5], [0.6, 0.6, 1e-1])

x0 = np.array([0.52, 0.47, 1e-3])
op = pints.OptimisationController(
    log_likelihood, x0, boundaries=boundaries, method=pints.CMAES
)
x1, f1 = op.run()
print("Estimated parameters:")
print(x1)


# Show the generated data
simulated_values = problem.evaluate(x1[:2])

plt.figure()
plt.xlabel("Time")
plt.ylabel("Values")
plt.plot(T, CorruptValues)
plt.fill_between(T, simulated_values - sigma, simulated_values + sigma, alpha=0.2)
plt.plot(T, simulated_values)
plt.show()
