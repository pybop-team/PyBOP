import numpy as np
import polars as pl
import pybamm

import pybop

"""
Example demonstrating coupled GITT-EIS parameterisation: a GITT experiment in which an
EIS spectrum is also acquired at the end of each pulse.

A synthetic dataset is built in two stages: a GITT experiment is simulated to obtain the
current as a function of time, then an impedance spectrum is computed about the state
reached at the end of each pulse. Both data types are stored in a single dataset on the
time domain: the voltage is recorded at every time, and the impedance takes a complex
value at each acquisition time and zero everywhere else.

Diffusivity and the exchange-current density are fitted together, which is the pairing
the two measurements are meant to separate: the relaxation after each pulse constrains
transport, while the charge-transfer semicircle of the spectrum constrains kinetics.
"""

# Define the model
model = pybamm.lithium_ion.SPM(
    options={"surface form": "differential", "contact resistance": "true"},
)
parameter_values = pybamm.ParameterValues("Chen2020")
parameter_values["Contact resistance [Ohm]"] = 0.01


# The exchange-current density of Chen2020 hard-codes its prefactor, so redefine it with
# the prefactor exposed as a parameter which can then be fitted
def positive_exchange_current_density(c_e, c_s_surf, c_s_max, T):
    m_ref = pybamm.Parameter(
        "Positive electrode reference exchange-current density [A.m-2]"
    )
    E_r = 17800
    arrhenius = pybamm.exp(E_r / pybamm.constants.R * (1 / 298.15 - 1 / T))
    return m_ref * arrhenius * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf) ** 0.5


parameter_values.update(
    {
        "Positive electrode reference exchange-current density [A.m-2]": 3.42e-6,
        "Positive electrode exchange-current density [A.m-2]": positive_exchange_current_density,
    }
)
parameter_values.set_initial_state(0.9)

# Simulate a GITT experiment: repeated pulses at 1C, each followed by a rest
n_pulses = 5
period = 5  # s
experiment = pybamm.Experiment(
    ["Discharge at 1C for 200 seconds", "Rest for 2400 seconds"] * n_pulses,
    period=f"{period} seconds",
)
gitt_solution = pybamm.Simulation(
    model, parameter_values=parameter_values, experiment=experiment
).solve()
dataset = pybop.import_pybamm_solution(gitt_solution)

# Acquire a spectrum at the end of each pulse, meaning at the end of each rest where the
# cell has relaxed and the current is zero
future_time = dataset["Time [s]"][-1] + 2 * period
acquisition_indices = (
    dataset.lf.with_row_index()
    .filter(pl.col("Current [A]") == 0)
    .filter(pl.col("Time [s]").shift(-1, fill_value=future_time).diff() > period)
    .select("index")
    .collect()
    .to_numpy()
    .ravel()
)

# Simulate the impedance about the state reached at the end of each pulse. This solves
# the protocol above once and linearises the model at each of the acquisition times
f_eval = np.logspace(-3, 4, 30)
solution = pybop.pybamm.EISSimulator(
    model,
    parameter_values=parameter_values,
    protocol=dataset,
    acquisition_indices=acquisition_indices,
    f_eval=f_eval,
).solve()

# Create the synthetic dataset with the simulated spectra, adding noise to both impedance
# and voltage. Only the acquired spectra carry noise; the remaining entries stay at zero
dataset = pybop.Dataset(
    {
        "Time [s]": solution["Time [s]"].data,
        "Current [A]": solution["Current [A]"].data,
        "Voltage [V]": pybop.add_noise(solution["Voltage [V]"].data, 1e-3),
        "Impedance [Ohm]": np.where(
            solution["Impedance [Ohm]"].data != 0,
            pybop.add_noise(solution["Impedance [Ohm]"].data, 1e-4),
            0.0,
        ),
    }
)

# Save the true values
true_values = [
    parameter_values[p]
    for p in [
        "Positive particle diffusivity [m2.s-1]",
        "Positive electrode reference exchange-current density [A.m-2]",
    ]
]

# Fitting parameters
parameter_values.update(
    {
        "Positive particle diffusivity [m2.s-1]": pybop.Parameter(
            pybop.LogUniform(1e-16, 1e-13),
            transformation=pybop.LogTransformation(),
        ),
        "Positive electrode reference exchange-current density [A.m-2]": pybop.Parameter(
            pybop.Uniform(1e-6, 1e-5)
        ),
    }
)

# Build the problem. The two error measures share the dataset and the time domain, so
# they can be combined with a weight setting their relative importance. Sum-based
# measures are used because the impedance variable is zero at most times, which would
# otherwise dilute the impedance term.
simulator = pybop.pybamm.EISSimulator(
    model,
    parameter_values=parameter_values,
    protocol=dataset,
    acquisition_indices=acquisition_indices,
    f_eval=f_eval,
)
voltage_cost = pybop.SumSquaredError(dataset, target=["Voltage [V]"])
impedance_cost = pybop.SumSquaredError(dataset, target=["Impedance [Ohm]"])
cost = pybop.WeightedCost(voltage_cost, impedance_cost, weights=[1.0, 1e3])
problem = pybop.Problem(simulator, cost)

# Set up the optimiser
options = pybop.PintsOptions(max_iterations=100, max_unchanged_iterations=25)
optim = pybop.XNES(problem, options=options)

# Run the optimisation
result = optim.run()
print(result)

# Compare identified to true parameter values
print("True parameters:", true_values)
print("Identified parameters:", result.x)

# Plot the optimisation result
problem.set_target("Voltage [V]")
pybop.plot.problem(problem, inputs=result.best_inputs, title="Optimised Comparison")
problem.set_target("Impedance [Ohm]")
pybop.plot.nyquist(problem, inputs=result.best_inputs, title="Optimised Comparison")
result.plot_convergence()
result.plot_parameters()
