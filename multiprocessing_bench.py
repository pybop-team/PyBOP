import cProfile
import os
import time

import numpy as np
import pybamm

import pybop

current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(
    current_dir, "examples/data/synthetic/dfn_charge_discharge_75.csv"
)

# Import the synthetic dataset
csv_data = np.loadtxt(dataset_path, delimiter=",", skiprows=1)
downsample = 1
dataset = pybop.Dataset(
    {
        "Time [s]": csv_data[::downsample, 0],
        "Current function [A]": csv_data[::downsample, 1],
        "Voltage [V]": csv_data[::downsample, 2],
        "Bulk open-circuit voltage [V]": csv_data[::downsample, 3],
    }
)

# Define model and parameter values
model = pybamm.lithium_ion.DFN()
parameter_values = pybamm.ParameterValues("Chen2020")
parameter_values.set_initial_state(f"{csv_data[0, 2]} V")

# Fitting parameters
parameter_values.update(
    {
        "Negative electrode active material volume fraction": pybop.Parameter(
            prior=pybop.Gaussian(0.68, 0.05),
            initial_value=0.65,
            bounds=[0.4, 0.9],
        ),
        "Positive electrode active material volume fraction": pybop.Parameter(
            prior=pybop.Gaussian(0.58, 0.05),
            initial_value=0.65,
            bounds=[0.4, 0.9],
        ),
    }
)

# Build the problem
simulator = pybop.pybamm.Simulator(model, parameter_values, protocol=dataset)
target = ["Voltage [V]", "Bulk open-circuit voltage [V]"]
cost = pybop.RootMeanSquaredError(dataset, target=target)
problem = pybop.Problem(simulator, cost)
logger = pybop.Logger(minimising=True)
pop_eval = pybop.PopulationEvaluator(
    problem, minimise=True, with_sensitivities=False, logger=logger
)
ser_eval = pybop.ScalarEvaluator(
    problem, minimise=True, with_sensitivities=False, logger=logger
)

N = 100
list_inputs = [
    problem.parameters.to_dict([(0.9 - 0.4) * i / N + 0.4, (0.9 - 0.4) * i / N + 0.4])
    for i in range(N)
]
t0 = time.perf_counter()
print(f"Running multiprocessing with {N} inputs...")
# Evaluate serially
t0 = time.perf_counter()
cProfile.runctx(
    "sols = [ser_eval._evaluate(inputs) for inputs in list_inputs]",
    globals(),
    locals(),
    filename="serial_profile.prof",
)
t_serial = time.perf_counter() - t0
# Evaluate in parallel

t0 = time.perf_counter()
cProfile.runctx(
    "sols_mp = pop_eval._evaluate(list_inputs)",
    globals(),
    locals(),
    filename="multiprocessing_profile.prof",
)
t_parallel = time.perf_counter() - t0

print(
    f"Serial time: {t_serial} s, Parallel time: {t_parallel} s, Speedup: {t_serial / t_parallel}"
)
