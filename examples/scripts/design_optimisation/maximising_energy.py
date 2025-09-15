import pybamm
from pybamm import Parameter

import pybop
from pybop.parameters.parameter_set import set_formation_concentrations

# A design optimisation example loosely based on work by L.D. Couto
# available at https://doi.org/10.1016/j.energy.2022.125966.

# The target is to maximise the energy density over a range of
# possible design parameter values, including for example:
# cross-sectional area = height x width (only need change one)
# electrode widths, particle radii, volume fractions and
# separator width.

# Define parameter set and additional parameters needed for the cost function
parameter_set = pybamm.ParameterValues("Chen2020")
set_formation_concentrations(parameter_set)
parameter_set.update(
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
    },
    check_already_exists=False,
)

# Define model
model = pybop.lithium_ion.SPMe(parameter_set=parameter_set)

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
signal = ["Voltage [V]", "Current [A]"]

# Generate problem
problem = pybop.DesignProblem(
    model,
    parameters,
    experiment,
    signal=signal,
    initial_state={"Initial SoC": 1.0},
    update_capacity=True,
)

# Generate multiple cost functions and combine them
cost1 = pybop.GravimetricEnergyDensity(problem)
cost2 = pybop.VolumetricEnergyDensity(problem)
cost = pybop.WeightedCost(cost1, cost2, weights=[1, 1e-3])

# Run optimisation
optim = pybop.PSO(
    cost, verbose=True, allow_infeasible_solutions=False, max_iterations=10
)
results = optim.run()
print(f"Initial gravimetric energy density: {cost1(optim.x0):.2f} Wh.kg-1")
print(f"Optimised gravimetric energy density: {cost1(results.x):.2f} Wh.kg-1")
print(f"Initial volumetric energy density: {cost2(optim.x0):.2f} Wh.m-3")
print(f"Optimised volumetric energy density: {cost2(results.x):.2f} Wh.m-3")

# Plot the timeseries output
pybop.plot.problem(problem, problem_inputs=results.x, title="Optimised Comparison")

# Plot the cost landscape with optimisation path
pybop.plot.surface(optim)
