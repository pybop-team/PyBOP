import numpy as np
import pybamm

import pybop

"""
Example demonstrating coupled GITT-EIS parameterisation: a GITT experiment in which an
EIS spectrum is also acquired at the end of each pulse.

A synthetic dataset is built in two stages: a GITT experiment is simulated to give the
time-domain voltage, then an impedance spectrum is computed about the state reached at
the end of each pulse. Both data types are stored in a single dataset on the time
domain: the voltage is recorded at every time, and the impedance variables hold the real
and imaginary components of each spectrum at the times of acquisition and zero
everywhere else.

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
    {"Positive electrode reference exchange-current density [A.m-2]": 3.42e-6},
    check_already_exists=False,
)
parameter_values["Positive electrode exchange-current density [A.m-2]"] = (
    positive_exchange_current_density
)
parameter_values.set_initial_state(0.9)

# Simulate a GITT experiment: repeated pulses at 1C, each followed by a rest
n_pulses = 5
pulse_duration = 300  # s
rest_duration = 2400  # s
period = 1.0  # s
experiment = pybamm.Experiment(
    [
        (
            f"Discharge at 1C for {pulse_duration} seconds",
            f"Rest for {rest_duration} seconds",
        )
    ]
    * n_pulses,
    period=f"{period} seconds",
)
gitt = pybamm.Simulation(
    model, parameter_values=parameter_values, experiment=experiment
).solve()

# Resample onto a uniform grid, avoiding the repeated times at each step change
time = np.arange(0, gitt.t[-1], period)
current = gitt["Current [A]"](time)
voltage = gitt["Voltage [V]"](time)

# Acquire a spectrum at the end of each pulse, meaning at the end of each rest where the
# cell has relaxed and the current is zero
f_eval = np.logspace(-3, 4, 30)
impedance_variables = pybop.get_impedance_variables(f_eval)
eis_times = [
    (i + 1) * (pulse_duration + rest_duration) - period for i in range(n_pulses)
]
eis_rows = [int(np.argmin(np.abs(time - t))) for t in eis_times]

# Assemble the dataset. The impedance variables start as a marker of which times were
# acquired, which is how the simulator learns where to compute a spectrum; the measured
# values replace the markers once they have been simulated below.
acquired = np.isin(np.arange(len(time)), eis_rows)
dataset = pybop.Dataset(
    {
        "Time [s]": time,
        "Current [A]": current,
        "Voltage [V]": voltage,
        **{name: acquired.astype(float) for name in impedance_variables},
    },
    domain="Time [s]",
)

# Simulate the impedance about the state reached at the end of each pulse. This solves
# the protocol above once and linearises the model at each of the acquisition times
sigma_v = 1e-3  # V
sigma_z = 1e-4  # Ohm
solution = pybop.pybamm.EISSimulator(
    model, parameter_values=parameter_values, protocol=dataset, f_eval=f_eval
).solve()

# Complete the synthetic dataset with the simulated spectra, adding noise to both data
# types. Only the acquired spectra carry noise; the remaining entries stay at zero
dataset["Voltage [V]"] = pybop.add_noise(voltage, sigma_v)
for name in impedance_variables:
    dataset[name] = np.where(
        acquired, pybop.add_noise(solution[name].data, sigma_z), 0.0
    )

# Save the true values
true_values = [
    parameter_values[p]
    for p in [
        "Positive particle diffusivity [m2.s-1]",
        "Positive electrode reference exchange-current density [A.m-2]",
    ]
]

# Fitting parameters, each searched over an order of magnitude around the true value
parameter_values.update(
    {
        "Positive particle diffusivity [m2.s-1]": pybop.Parameter(
            pybop.Uniform(1e-16, 1e-13)
        ),
        "Positive electrode reference exchange-current density [A.m-2]": pybop.Parameter(
            pybop.Uniform(1e-6, 1e-5)
        ),
    }
)

# Build the problem. The two error measures share the dataset and the time domain, so
# they can be combined with a weight setting their relative importance. Sum-based
# measures are used because the impedance variables are zero at most times, which would
# otherwise dilute the impedance term.
simulator = pybop.pybamm.EISSimulator(
    model, parameter_values=parameter_values, protocol=dataset, f_eval=f_eval
)
voltage_cost = pybop.SumSquaredError(dataset, target=["Voltage [V]"])
impedance_cost = pybop.SumSquaredError(dataset, target=impedance_variables)
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
pybop.plot.problem(problem, inputs=result.best_inputs, title="Optimised Comparison")
result.plot_convergence()
result.plot_parameters()
