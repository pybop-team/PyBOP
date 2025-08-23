import pybamm
from pybamm import Parameter

import pybop

# Define model, parameter values and additional parameters needed for the cost function
model = pybamm.lithium_ion.SPM()
parameter_values = pybamm.ParameterValues("Marquis2019")
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

# Target parameters
parameters = [
    pybop.Parameter(
        "Positive electrode thickness [m]",
        initial_value=9e-05,
        bounds=[6.5e-05, 12e-05],
    ),
    pybop.Parameter(
        "Negative electrode thickness [m]",
        initial_value=9e-05,
        bounds=[5e-05, 12e-05],
    ),
]


# Define test protocol
experiment = pybamm.Experiment(
    [
        "Discharge at 3C until 3.0 V (2 minute period)",
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
    .add_cost(pybop.costs.pybamm.GravimetricPowerDensity())
    .add_cost(pybop.costs.pybamm.VolumetricPowerDensity(), weight=1e-4)
)
for param in parameters:
    builder.add_parameter(param)

problem = builder.build()

# Set optimiser and options
options = pybop.SciPyDifferentialEvolutionOptions(
    verbose=True,
    maxiter=5,
    polish=False,
)
optim = pybop.SciPyDifferentialEvolution(problem, options=options)
results = optim.run()

# Obtain the identified pybamm.ParameterValues object for use with PyBaMM classes
identified_parameter_values = results.parameter_values

# Plot convergence
pybop.plot.convergence(optim)

# Plot the parameter traces
pybop.plot.parameters(optim)

# Plot the cost landscape with optimisation path
pybop.plot.surface(optim)

print(f"Initial power density: {-results.initial_cost:.2f} Wh.kg-1")
print(f"Optimised power density: {-results.best_cost:.2f} Wh.kg-1")
