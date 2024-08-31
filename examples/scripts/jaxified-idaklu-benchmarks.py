import time

import jax
import jax.numpy as jnp
import numpy as np
import pybamm

import pybop

n = 30  # Number of solves
output_vars = [
    "Voltage [V]",
    "Current [A]",
    "Time [s]",
]

solvers = [
    pybamm.CasadiSolver(mode="fast with events", atol=1e-6, rtol=1e-6),
    pybamm.IDAKLUSolver(atol=1e-6, rtol=1e-6, output_variables=output_vars),
]

# Parameter set and model definition
parameter_set = pybop.ParameterSet.pybamm("Chen2020")
model = pybop.lithium_ion.DFN(parameter_set=parameter_set, solver=solvers[0])

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
t_eval = np.linspace(0, 100, 1000)
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


# Create inputs function for benchmarking
def inputs():
    return {
        "Negative electrode active material volume fraction": 0.55
        + np.random.normal(0, 0.01),
        "Positive electrode active material volume fraction": 0.55
        + np.random.normal(0, 0.01),
    }


# Iterate over the solvers and print benchmarks
for solver in solvers:
    # Setup Fitting Problem
    model.solver = solver
    problem = pybop.FittingProblem(model, parameters, dataset)
    cost = pybop.SumSquaredError(problem)

    start_time = time.time()
    for _i in range(n):
        out = problem.model.simulate(inputs=inputs(), t_eval=t_eval)
    print(f"({solver.name}) Time model.simulate: {time.time() - start_time:.4f}")

    start_time = time.time()
    for _i in range(n):
        out = problem.model.simulateS1(inputs=inputs(), t_eval=t_eval)
    print(f"({solver.name}) Time model.SimulateS1: {time.time() - start_time:.4f}")

    start_time = time.time()
    for _i in range(n):
        out = problem.evaluate(inputs=inputs())
    print(f"({solver.name}) Time problem.evaluate: {time.time() - start_time:.4f}")

    start_time = time.time()
    for _i in range(n):
        out = problem.evaluateS1(inputs=inputs())
    print(f"({solver.name}) Time Problem.EvaluateS1: {time.time() - start_time:.4f}")

    start_time = time.time()
    for _i in range(n):
        out = cost(inputs(), calculate_grad=False)
    print(f"({solver.name}) Time PyBOP Cost w/o grad: {time.time() - start_time:.4f}")

    start_time = time.time()
    for _i in range(n):
        out = cost(inputs(), calculate_grad=True)
    print(f"({solver.name}) Time PyBOP Cost w/grad: {time.time() - start_time:.4f}")

# Jaxified benchmarks
ida_solver = pybamm.IDAKLUSolver(atol=1e-6, rtol=1e-6, output_variables=output_vars)
solver = ida_solver.jaxify(model=model.built_model, t_eval=t_eval)
solver_no_grad = ida_solver.jaxify(
    model=model.built_model, t_eval=t_eval, calculate_sensitivities=False
)
f = solver.get_jaxpr()
k = solver_no_grad.get_jaxpr()

start_time = time.time()
for _i in range(n):
    kout = k(t_eval, inputs())
print(f"Time jax expression w/o grad: {time.time() - start_time:.4f}")

start_time = time.time()
for _i in range(n):
    fout = f(t_eval, inputs())
print(f"Time jax expression w/ grad: {time.time() - start_time:.4f}")

start_time = time.time()
for _i in range(n):
    y = solver_no_grad.get_var("Voltage [V]")(t_eval, inputs())
print(f"Time jax solver_no_grad.get_var: {time.time() - start_time:.4f}")

start_time = time.time()
for _i in range(n):
    y = solver.get_var("Voltage [V]")(t_eval, inputs())
print(f"Time jax solver.get_var: {time.time() - start_time:.4f}")


# Sum-of-squared errors
def sse_no_grad(t_eval, inputs):
    y = solver_no_grad.get_var("Voltage [V]")(t_eval, inputs)
    r = jnp.asarray([y - problem.target[signal] for signal in problem.signal])
    return jnp.sum(jnp.sum(r**2, axis=0), axis=0)


# Sum-of-squared errors
def sse_grad(t_eval, inputs):
    y = solver.get_var("Voltage [V]")(t_eval, inputs)
    r = jnp.asarray([y - problem.target[signal] for signal in problem.signal])
    return jnp.sum(jnp.sum(r**2, axis=0), axis=0)


start_time = time.time()
for _i in range(n):
    out = sse_no_grad(t_eval, inputs())
print(f"Time Jax SumSquaredError w/o grad: {time.time() - start_time:.4f}")

start_time = time.time()
for _i in range(n):
    out = jax.value_and_grad(sse_grad, argnums=1)(t_eval, inputs())
print(f"Time Jax SumSquaredError w/ grad: {time.time() - start_time:.4f}")
