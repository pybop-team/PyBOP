import time

import numpy as np
import pybamm

import pybop

# Parameter set and model definition
parameter_set = pybop.ParameterSet.pybamm("Chen2020")
solver = pybamm.IDAKLUSolver(atol=1e-6, rtol=1e-6)
model = pybop.lithium_ion.SPMe(parameter_set=parameter_set, solver=solver)

solvers = [
    pybamm.IDAKLUSolver(atol=1e-6, rtol=1e-6),
    pybamm.CasadiSolver(atol=1e-6, rtol=1e-6),
]

for solver in solvers:
    model.build()
    model.solver = solver
    t1 = time.time()
    model.predict(t_eval=np.linspace(0, 1, 100))
    print(time.time() - t1)

# # Fitting parameters
# parameters = pybop.Parameters(
#     pybop.Parameter(
#         "Negative electrode active material volume fraction",
#         prior=pybop.Gaussian(0.68, 0.05),
#     ),
#     pybop.Parameter(
#         "Positive electrode active material volume fraction",
#         prior=pybop.Gaussian(0.58, 0.05),
#     ),
# )
#
# # Generate data
# sigma = 0.001
# t_eval = np.linspace(0, 900, 1800)
# values = model.predict(t_eval=t_eval)
# corrupt_values = values["Voltage [V]"].data + np.random.normal(0, sigma, len(t_eval))
#
# # Form dataset
# dataset = pybop.Dataset(
#     {
#         "Time [s]": t_eval,
#         "Current function [A]": values["Current [A]"].data,
#         "Voltage [V]": corrupt_values,
#     }
# )
#
# # Generate problem, cost function, and optimisation class
# problem = pybop.FittingProblem(model, parameters, dataset)
# cost = pybop.SumSquaredError(problem)
# # optim = pybop.Adam(
# #     cost,
# #     sigma0=0.0001,
# #     verbose=True,
# #     min_iterations=125,
# #     max_iterations=125,
# # )
#
# # Run optimisation
# # t1 = time.time()
# # x, final_cost = optim.run()
# # t2 = time.time()
# # print(f"Time:{t2-t1}")
# # print("Estimated parameters:", x)
#
# # Create the list of input dicts
# n = 150  # Number of solves
# inputs = [[x, y] for x, y in zip(np.linspace(0.45, 0.6, n), np.linspace(0.45, 0.6, n))]
#
# t1 = time.time()
# for i in range(n):
#     k = model.simulate(t_eval=t_eval, inputs=inputs[i])
# t2 = time.time()
# print(f"Time Evaluate:{t2-t1}")
#
# t1 = time.time()
# for i in range(n):
#     k = model.simulateS1(t_eval=t_eval, inputs=inputs[i])
# t2 = time.time()
# print(f"Time EvaluateS1:{t2-t1}")
