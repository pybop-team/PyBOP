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

# Define parameter set and additional parameters needed for the cost function
parameter_values = pybamm.ParameterValues("Chen2020")
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
    },
    check_already_exists=False,
)

# Define model
model = pybamm.lithium_ion.DFN()

# Fitting parameters
parameters = [
    pybop.Parameter(
        "Positive electrode thickness [m]",
        prior=pybop.Gaussian(9e-05, 0.1e-05),
        transformation=pybop.LogTransformation(),
        bounds=[6.5e-05, 12e-05],
    ),
    pybop.Parameter(
        "Negative electrode thickness [m]",
        prior=pybop.Gaussian(9e-05, 0.1e-05),
        transformation=pybop.LogTransformation(),
        bounds=[5e-05, 12e-05],
    ),
]

# Define test protocol
experiment = pybamm.Experiment(
    [
        "Discharge at 3C until 2.5 V (2 minute period)",
    ],
)

# Construct the problem builder
builder = (
    pybop.builders.Pybamm()
    .set_simulation(
        model,
        parameter_values=parameter_values,
        experiment=experiment,
        initial_state=1.0,
    )
    .add_cost(pybop.costs.pybamm.GravimetricEnergyDensity())
    .add_cost(pybop.costs.pybamm.VolumetricEnergyDensity(), weight=1e-4)
)
for param in parameters:
    builder.add_parameter(param)
problem = builder.build()

# Set optimiser and options
options = pybop.PintsOptions(
    verbose=True, max_iterations=15, max_unchanged_iterations=10
)
optim = pybop.CMAES(problem, options=options)
results = optim.run()

# Obtain the fully identified pybamm.ParameterValues object
# These can then be used with normal Pybamm classes
# identified_parameter_values = results.parameter_values

# Plot convergence
pybop.plot.convergence(optim)

# Plot the parameter traces
pybop.plot.parameters(optim)

# Plot the cost landscape with optimisation path
pybop.plot.surface(optim)

# results = optim.run()
# print(f"Initial gravimetric energy density: {cost1(optim.x0):.2f} Wh.kg-1")
# print(f"Optimised gravimetric energy density: {cost1(results.x):.2f} Wh.kg-1")
# print(f"Initial volumetric energy density: {cost2(optim.x0):.2f} Wh.m-3")
# print(f"Optimised volumetric energy density: {cost2(results.x):.2f} Wh.m-3")
