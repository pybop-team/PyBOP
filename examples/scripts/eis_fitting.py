import numpy as np
import plotly.express as px

import pybop

# Define model
parameter_set = pybop.ParameterSet.pybamm("Chen2020")
model = pybop.lithium_ion.DFN(
    parameter_set=parameter_set, options={"surface form": "differential"}
)

# Fitting parameters
parameters = pybop.Parameters(
    pybop.Parameter(
        "Positive electrode double-layer capacity [F.m-2]",
        prior=pybop.Gaussian(0.1, 0.05),
    ),
    pybop.Parameter(
        "Negative electrode thickness [m]",
        prior=pybop.Gaussian(40e-6, 0.0),
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
prediction_1 = problem.evaluate(np.array([1.0, 60e-6]))
prediction_2 = problem.evaluate(np.array([10.0, 40e-6]))
fig = px.scatter(x=prediction_1["Impedance"].real, y=-prediction_1["Impedance"].imag)
fig.add_scatter(x=prediction_2["Impedance"].real, y=-prediction_2["Impedance"].imag)
fig.show()
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
