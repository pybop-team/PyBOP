import numpy as np
import plotly.express as px

import pybop

# Define model
parameter_set = pybop.ParameterSet.pybamm("Chen2020")
model = pybop.lithium_ion.SPM(
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
prediction_1 = problem.evaluate(np.array([0.1, 50e-6]))
prediction_2 = problem.evaluate(np.array([10, 70e-6]))

# Plot
fig = px.scatter(x=prediction_1["Impedance"].real, y=-prediction_1["Impedance"].imag)
fig.add_scatter(x=prediction_2["Impedance"].real, y=-prediction_2["Impedance"].imag)
fig.show()
