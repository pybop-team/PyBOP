import time

import numpy as np
import pybamm

import pybop

# Parameter set and model definition
parameter_set = pybop.ParameterSet.pybamm("Chen2020")
model = pybop.lithium_ion.SPM(parameter_set=parameter_set)

solvers = [
    pybamm.IDAKLUSolver(atol=1e-6, rtol=1e-6),
    pybamm.CasadiSolver(mode="safe", atol=1e-6, rtol=1e-6),
    pybamm.CasadiSolver(mode="fast with events", atol=1e-6, rtol=1e-6),
]

# Fitting parameters
parameters = pybop.Parameters(
    pybop.Parameter(
        "Negative electrode active material volume fraction", initial_value=0.55
    ),
    pybop.Parameter(
        "Positive electrode active material volume fraction", initial_value=0.55
    ),
)

# Define test protocol and generate data
experiment = pybop.Experiment([("Discharge at 0.5C for 10 minutes (3 second period)")])
values = model.predict(
    initial_state={"Initial open-circuit voltage [V]": 4.2}, experiment=experiment
)

# Form dataset
dataset = pybop.Dataset(
    {
        "Time [s]": values["Time [s]"].data,
        "Current function [A]": values["Current [A]"].data,
        "Voltage [V]": values["Voltage [V]"].data,
    }
)

# Create the list of input dicts
n = 150  # Number of solves
inputs = list(zip(np.linspace(0.45, 0.6, n), np.linspace(0.45, 0.6, n)))

# Iterate over the solvers and print benchmarks
for solver in solvers:
    model.solver = solver
    problem = pybop.FittingProblem(model, parameters, dataset)

    start_time = time.time()
    for input_values in inputs:
        problem.evaluate(inputs=input_values)
    print(f"Time Evaluate {solver.name}: {time.time() - start_time:.3f}")

    start_time = time.time()
    for input_values in inputs:
        problem.evaluateS1(inputs=input_values)
    print(f"Time EvaluateS1 {solver.name}: {time.time() - start_time:.3f}")
