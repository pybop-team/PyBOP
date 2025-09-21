import pybamm
from pybamm import Parameter

import pybop
from pybop.pybamm.parameter_utils import set_formation_concentrations

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
set_formation_concentrations(parameter_values)
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
parameters = pybop.Parameters(
    pybop.Parameter(
        "Positive electrode thickness [m]",
        prior=pybop.Gaussian(7.56e-05, 0.1e-05),
        bounds=[65e-06, 10e-05],
    ),
    pybop.Parameter(
        "Positive particle radius [m]",
        prior=pybop.Gaussian(5.22e-06, 0.1e-06),
        bounds=[2e-06, 9e-06],
    ),
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
    input_parameter_names=parameters.names,
    protocol=experiment,
    initial_state={"Initial SoC": 1.0},
)
problem = pybop.DesignProblem(
    simulator,
    parameters,
    output_variables=[
        "Voltage [V]",
        "Gravimetric energy density [Wh.kg-1]",
        "Volumetric energy density [Wh.m-3]",
    ],
)

# Generate multiple cost functions and combine them
cost1 = pybop.DesignCost(problem, target="Gravimetric energy density [Wh.kg-1]")
cost2 = pybop.DesignCost(problem, target="Volumetric energy density [Wh.m-3]")
cost = pybop.WeightedCost(cost1, cost2, weights=[1, 1e-3])

# Run the optimisation
options = pybop.PintsOptions(max_iterations=10)
optim = pybop.PSO(cost, options=options)
result = optim.run()
print(result)
print(f"Initial gravimetric energy density: {cost1(result.x0):.2f} Wh.kg-1")
print(f"Optimised gravimetric energy density: {cost1(result.x):.2f} Wh.kg-1")
print(f"Initial volumetric energy density: {cost2(result.x0):.2f} Wh.m-3")
print(f"Optimised volumetric energy density: {cost2(result.x):.2f} Wh.m-3")

# Plot the timeseries output
pybop.plot.problem(problem, problem_inputs=result.x, title="Optimised Comparison")

# Plot the optimisation result
result.plot_surface()
