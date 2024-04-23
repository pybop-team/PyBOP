import timeit

import numpy as np
import plotly.graph_objects as go

import pybop

# Define model
parameter_set = pybop.ParameterSet.pybamm("Chen2020")
model = pybop.lithium_ion.SPM(parameter_set=parameter_set)

# Fitting parameters
parameters = [
    pybop.Parameter(
        "Negative electrode active material volume fraction",
        prior=pybop.Gaussian(0.6, 0.05),
        bounds=[0.5, 0.8],
    ),
    pybop.Parameter(
        "Positive electrode active material volume fraction",
        prior=pybop.Gaussian(0.48, 0.05),
        bounds=[0.4, 0.7],
    ),
]

sigma = 0.001
t_eval = np.arange(0, 900, 2)
values = model.predict(t_eval=t_eval)
corrupt_values = values["Voltage [V]"].data + np.random.normal(0, sigma, len(t_eval))

# Form dataset
dataset = pybop.Dataset(
    {
        "Time [s]": t_eval,
        "Current function [A]": values["Current [A]"].data,
        "Voltage [V]": corrupt_values,
    }
)


def run_optim(optim):
    x, final_cost = optim.run()


def run_optims(multi, cores, iters, num_runs):
    times = []
    for core in cores:
        problem = pybop.FittingProblem(model, parameters, dataset)
        cost = pybop.SumSquaredError(problem)
        optim = pybop.Optimisation(cost, optimiser=pybop.CMAES)
        optim.optimiser.set_population_size(core)
        if multi:
            optim.set_parallel(True)
        optim.set_min_iterations(iters)
        optim.set_max_iterations(iters)
        # Create a Timer object for the current core configuration
        timer = timeit.Timer(lambda: run_optim(optim))

        # Run the benchmark multiple times (e.g., 3 times) and get the execution times
        execution_times = timer.repeat(repeat=num_runs, number=1)

        # Append the execution times to times_multi
        times.append(execution_times)
    return times


population_size = [2, 4, 8, 12]
times_multi = []
times_multi = run_optims(True, population_size, 1000, 1)
times_single = run_optims(False, population_size, 1000, 1)

times_multi_mean = np.mean(times_multi, axis=1)
times_single_mean = np.mean(times_single, axis=1)

fig = go.Figure(
    data=[
        go.Scatter(
            x=population_size,
            y=times_multi_mean,
            mode="lines",
            showlegend=True,
            name="Multiprocessing",
            # marker=dict(size=5, color="blue"),
        ),
        go.Scatter(
            x=population_size,
            y=times_single_mean,
            mode="lines",
            showlegend=True,
            name="Single Processing",
            # marker=dict(size=5, color="red"),
        ),
    ],
    layout=go.Layout(
        title="",
        height=600,
        width=800,
        plot_bgcolor="white",
        font=dict(size=20),
        yaxis_title="Time / s",
        xaxis_title="Population Size / n",
    ),
)

# fig.show()
fig.write_image("multiprocessing.pdf")
fig.write_image("multiprocessing.pdf")
