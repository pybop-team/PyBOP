import pybamm
from pybamm import Parameter

import pybop

# A design optimisation example loosely based on work by L.D. Couto
# available at https://doi.org/10.1016/j.energy.2022.125966.

# The target is to maximise the energy density over a range of
# possible design parameter values, including for example:
# cross-sectional area = height x width (only need change one)
# electrode widths, particle radii, volume fractions and
# separator width.

# Define model
model = pybamm.lithium_ion.SPMe()
pybop.pybamm.add_variable_to_model(model, "Gravimetric energy density [Wh.kg-1]")
pybop.pybamm.add_variable_to_model(model, "Volumetric energy density [Wh.m-3]")

# Define parameter set and additional parameters needed for the cost function
parameter_values = pybamm.ParameterValues("Chen2020")
pybop.pybamm.set_formation_concentrations(parameter_values)
parameter_values.update(
    {
        "Electrolyte density [kg.m-3]": Parameter("Separator density [kg.m-3]"),
        "Negative electrode active material density [kg.m-3]": Parameter(
            "Negative electrode density [kg.m-3]"
        ),
        "Negative electrode carbon-binder density [kg.m-3]": Parameter(
            "Negative electrode density [kg.m-3]"
        ),
        "Positive electrode active material density [kg.m-3]": Parameter(
            "Positive electrode density [kg.m-3]"
        ),
        "Positive electrode carbon-binder density [kg.m-3]": Parameter(
            "Positive electrode density [kg.m-3]"
        ),
        "Cell mass [kg]": pybop.pybamm.cell_mass(),
        "Cell volume [m3]": pybop.pybamm.cell_volume(),
    },
    check_already_exists=False,
)

# Fitting parameters
parameter_values.update(
    {
        "Positive electrode thickness [m]": pybop.Parameter(
            "Positive electrode thickness [m]",
            prior=pybop.Gaussian(7.56e-05, 0.1e-05),
            bounds=[65e-06, 10e-05],
        ),
        "Positive particle radius [m]": pybop.Parameter(
            "Positive particle radius [m]",
            prior=pybop.Gaussian(5.22e-06, 0.1e-06),
            bounds=[2e-06, 9e-06],
        ),
    }
)

# Define test protocol
experiment = pybamm.Experiment(
    [
        "Discharge at 1C until 2.5 V (10 seconds period)",
        "Hold at 2.5 V for 30 minutes or until 10 mA (10 seconds period)",
    ],
)

# Generate problem
simulator = pybop.pybamm.Simulator(
    model,
    parameter_values=parameter_values,
    protocol=experiment,
    initial_state={"Initial SoC": 1.0},
)
cost_1 = pybop.DesignCost(target="Gravimetric energy density [Wh.kg-1]")
problem_1 = pybop.Problem(simulator, cost_1)
cost_2 = pybop.DesignCost(target="Volumetric energy density [Wh.m-3]")
problem_2 = pybop.Problem(simulator, cost_2)
problem = pybop.MetaProblem(problem_1, problem_2, weights=[1, 1e-3])

# Run the optimisation
options = pybop.PintsOptions(max_iterations=10)
optim = pybop.PSO(problem, options=options)
result = optim.run()
print(result)
print(f"Initial gravimetric energy density: {problem_1(result.x0):.2f} Wh.kg-1")
print(f"Optimised gravimetric energy density: {problem_1(result.x):.2f} Wh.kg-1")
print(f"Initial volumetric energy density: {problem_2(result.x0):.2f} Wh.m-3")
print(f"Optimised volumetric energy density: {problem_2(result.x):.2f} Wh.m-3")

# Plot the optimisation result
result.plot_surface()

# Plot the timeseries output
problem_1.target = "Voltage [V]"
pybop.plot.problem(problem_1, problem_inputs=result.x, title="Optimised Comparison")
