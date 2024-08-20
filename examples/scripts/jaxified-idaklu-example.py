import numpy as np
import pybamm

import pybop

# Parameter set and model definition
parameter_set = pybop.ParameterSet.pybamm("Chen2020")
model = pybop.lithium_ion.SPM(parameter_set=parameter_set)

output_vars = [
    "Voltage [V]",
    "Current [A]",
    "Time [s]",
]
solvers = [
    pybamm.IDAKLUSolver(atol=1e-6, rtol=1e-6, output_variables=output_vars),
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
# experiment = pybop.Experiment([("Discharge at 0.5C for 10 minutes (3 second period)")])
t_eval = np.linspace(0, 10, 100)
values = model.predict(
    initial_state={"Initial open-circuit voltage [V]": 4.2}, t_eval=t_eval
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
    model.build(
        inputs={
            "Negative electrode active material volume fraction": 0.55,
            "Positive electrode active material volume fraction": 0.55,
        }
    )
    solver = solver.jaxify(model=model.built_model, t_eval=t_eval)
    f = solver.get_jaxpr()
    print(f"JAX expression: {f}")
    model.solver = f

    # Setup Fitting Problem
    problem = pybop.JaxFittingProblem(model, parameters, dataset)
    problem.evaluate([0.55, 0.55])

    # start_time = time.time()
    # for input_values in inputs:
    #     problem.evaluate(inputs=input_values)
    # print(f"Time Evaluate {solver.name}: {time.time() - start_time:.3f}")

    # start_time = time.time()
    # for input_values in inputs:
    #     problem.evaluateS1(inputs=input_values)
    # print(f"Time EvaluateS1 {solver.name}: {time.time() - start_time:.3f}")