import matplotlib.pyplot as plt
import numpy as np
import pybamm

import pybop

# Generate a synthetic dataset to fit to. When working with an
# experiment we wouldn't need to generate synthetic data, but here it
# gives us a known ground-truth to work to.
model = pybamm.equivalent_circuit.Thevenin()
parameters = pybamm.ParameterValues(
    {
        "Initial SoC": 0.75,
        "Entropic change [V/K]": 0,
        "Cell thermal mass [J/K]": np.inf,
        "Cell-jig heat transfer coefficient [W/K]": 0,
        "Jig thermal mass [J/K]": np.inf,
        "Jig-air heat transfer coefficient [W/K]": 0,
        "Ambient temperature [K]": 25,
        "Initial temperature [K]": 25,
        "Cell capacity [A.h]": 5,
        "Nominal cell capacity [A.h]": 5,
        "Current function [A]": 5,
        "Upper voltage cut-off [V]": 4.2,
        "Lower voltage cut-off [V]": 3.0,
        "Open-circuit voltage [V]": pybamm.equivalent_circuit.Thevenin().default_parameter_values[
            "Open-circuit voltage [V]"
        ],
        "R0 [Ohm]": 0.001,
        "R1 [Ohm]": 0.002,
        "C1 [F]": 10000,
        "Element-1 initial overpotential [V]": 0,
    }
)
experiment = pybamm.Experiment(
    [
        "Rest for 1 minute",
        "Discharge at 1C for 2 minutes",
        "Charge at 1C for 1 minutes",
        "Rest for 1 minute",
    ]
)
sim = pybamm.Simulation(model=model, parameter_values=parameters, experiment=experiment)
sol = sim.solve()
sigma = 0.001
corrupt_values = sol["Voltage [V]"].data + np.random.normal(0, sigma, len(sol.t))
fitting_data = pybop.Dataset(
    {
        "Time [s]": sol.t,
        "Current function [A]": sol["Current [A]"].data,
        "Voltage [V]": corrupt_values,
    }
)


# Now, let's pretend we never did any of the previous, and we're
# trying to fit a 1-RC model to some data (fitting_data) taken from an
# experiment. First, we define the model to fit, then tell PyBOP about
# the necessary parameters, cost func, etc., then optimise!

# Set up the model
model = pybamm.equivalent_circuit.Thevenin()

# This contains both known and unknown parameters. Here, we're trying
# to fit R0, R1, C1, We don't know what the right values are for them
# yet, but we need to define them anyway. Let's start with a guess
# that's close to, but different from, the true values our data were
# generated with. Note how R0, R1, C1 differ from previously.
parameters = pybamm.ParameterValues(
    {
        "Initial SoC": 0.75,
        "Entropic change [V/K]": 0,
        "Cell thermal mass [J/K]": np.inf,
        "Cell-jig heat transfer coefficient [W/K]": 0,
        "Jig thermal mass [J/K]": np.inf,
        "Jig-air heat transfer coefficient [W/K]": 0,
        "Ambient temperature [K]": 25,
        "Initial temperature [K]": 25,
        "Cell capacity [A.h]": 5,
        "Nominal cell capacity [A.h]": 5,
        "Current function [A]": 5,
        "Upper voltage cut-off [V]": 4.2,
        "Lower voltage cut-off [V]": 3.0,
        "Open-circuit voltage [V]": pybamm.equivalent_circuit.Thevenin().default_parameter_values[
            "Open-circuit voltage [V]"
        ],
        "R0 [Ohm]": 0.0008,
        "R1 [Ohm]": 0.001,
        "C1 [F]": 10150,
        "Element-1 initial overpotential [V]": 0,
    }
)

# PyBaMM wants to see capacitances, but it's better to fit
# time-constants, so let's introduce some parameters to enable that
parameters.update(
    {
        "tau1 [s]": parameters["R1 [Ohm]"] * parameters["C1 [F]"],
    },
    check_already_exists=False,
)
parameters.update(
    {
        "C1 [F]": pybamm.Parameter("tau1 [s]") / pybamm.Parameter("R1 [Ohm]"),
    }
)

# Now we build a problem, and run the optimiser on it
builder = pybop.builders.Pybamm()
builder.set_dataset(fitting_data)
builder.set_simulation(model, parameter_values=parameters)
builder.add_parameter(
    pybop.Parameter(
        "R0 [Ohm]",
        initial_value=0.0008,
    )
)
builder.add_parameter(
    pybop.Parameter(
        "R1 [Ohm]",
        initial_value=0.001,
    )
)
builder.add_parameter(
    pybop.Parameter(
        "tau1 [s]",
        initial_value=10.15,
    )
)
builder.add_cost(
    pybop.costs.pybamm.SumSquaredError("Voltage [V]", "Voltage [V]"),
)
problem = builder.build()

optimiser = pybop.SciPyMinimize(
    problem, pybop.ScipyMinimizeOptions(method="Nelder-Mead")
)
fit = optimiser.run()
print(fit)


# Plot fitted result!
problem.set_params(fit.x)
fitsol = problem.pipeline.solve()
_, ax = plt.subplots()
ax.plot(sol.t, sol["Voltage [V]"].data, label="Target")
ax.plot(fitsol.t, fitsol["Voltage [V]"].data, label="Fit")
ax.legend()
ax.set_xlabel("Time [s]")
ax.set_ylabel("Voltage [V]")
plt.show()
