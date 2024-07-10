import numpy as np

import pybop

# Define model
parameter_set = pybop.ParameterSet.pybamm("Chen2020")
model = pybop.lithium_ion.DFN(
    parameter_set=parameter_set, options={"surface form": "differential"}
)

# Fitting parameters
parameters = pybop.Parameters(
    pybop.Parameter(
        "Negative electrode active material volume fraction",
        prior=pybop.Gaussian(0.6, 0.05),
    ),
    pybop.Parameter(
        "Positive electrode active material volume fraction",
        prior=pybop.Gaussian(0.48, 0.05),
    ),
)

# Generate data
sigma = 0.001
t_eval = np.arange(0, 900, 3)
values = model.predict(t_eval=t_eval)
corrupt_values = values["Voltage [V]"].data + np.random.normal(0, sigma, len(t_eval))

# Form dataset
dataset = pybop.Dataset(
    {
        "Frequency [Hz]": np.logspace(-4, 5, 300),
        "Current function [A]": np.ones(300) * 0.0,
        "Impedance": np.ones(300),
    }
)

signal = ["Impedance"]
# Generate problem, cost function, and optimisation class
problem = pybop.EISProblem(model, parameters, dataset, signal=signal)
prediction = problem.evaluate(np.array([0.75, 0.665]))
# fig = px.scatter(x=prediction["Impedance"].real, y=-prediction["Impedance"].imag)
# fig.show()
# cost = pybop.SumSquaredError(problem)
# optim = pybop.CMAES(cost, max_iterations=100)

# # Run the optimisation
# x, final_cost = optim.run()
# print("True parameters:", parameters.true_value())
# print("Estimated parameters:", x)

# # Plot the time series
# pybop.plot_dataset(dataset)

# # Plot the timeseries output
# pybop.quick_plot(problem, problem_inputs=x, title="Optimised Comparison")

# # Plot convergence
# pybop.plot_convergence(optim)

# # Plot the parameter traces
# pybop.plot_parameters(optim)

# # Plot the cost landscape
# pybop.plot2d(cost, steps=15)

# # Plot the cost landscape with optimisation path
# pybop.plot2d(optim, steps=15)
