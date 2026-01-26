import pybamm

import pybop

# Define a test protocol
init_soc = 0.5
experiment = pybamm.Experiment(
    [
        "Rest for 100 seconds",
        (
            "Charge at 1C for 100 seconds",
            "Rest for 100 seconds",
            "Discharge at 1C for 100 seconds",
            "Rest for 100 seconds",
        )
        * 10,
        "Rest for 1000 seconds",
    ]
)

# Generate synthetic data (no thermal model)
parameter_values = pybamm.ParameterValues("Chen2020")
model = pybamm.lithium_ion.SPM()
solution = pybamm.Simulation(
    model, parameter_values=parameter_values, experiment=experiment
).solve(initial_soc=init_soc)

# Add thermal parameters and the voltage data
parameter_values.update(
    {
        "Cell thermal mass [J/K]": 20,
        "Heat transfer coefficient [W/K]": 0.05,
        "Current function [A]": pybamm.Interpolant(
            solution.t, solution["Current [A]"].data, pybamm.t
        ),
        "Voltage function [V]": pybamm.Interpolant(
            solution.t, solution["Voltage [V]"].data, pybamm.t
        ),
    },
    check_already_exists=False,
)

# Group the parameters and relate the entropic coefficients to the cell-level
grouped_parameter_values = pybop.lithium_ion.CellTemperature.create_grouped_parameters(
    parameter_values
)
grouped_parameter_values.update(
    {"OCV entropic change [V.K-1]": 2e-5}, check_already_exists=False
)
grouped_parameter_values.update(
    {
        "Negative electrode OCP entropic change [V.K-1]": (
            -pybamm.Parameter("OCV entropic change [V.K-1]") / 2
        ),
        "Positive electrode OCP entropic change [V.K-1]": (
            pybamm.Parameter("OCV entropic change [V.K-1]") / 2
        ),
    }
)

# Run an example thermal simulation
thermal_model = pybop.lithium_ion.CellTemperature()
solution = pybamm.Simulation(
    thermal_model, parameter_values=grouped_parameter_values
).solve(initial_soc=init_soc, t_eval=solution.t)
solution.plot(["Current [A]", "SoC", "Voltage [V]", "Cell temperature [K]"])
