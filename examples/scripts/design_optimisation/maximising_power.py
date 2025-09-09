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
    .add_cost(pybop.costs.pybamm.VolumetricPowerDensity())
)
for param in parameters:
    builder.add_parameter(param)

problem = builder.build()

# Set optimiser and options. Restrict the maximum number of iterations for speed
options = pybop.SciPyDifferentialEvolutionOptions(
    maxiter=5,
    polish=False,
)
optim = pybop.SciPyDifferentialEvolution(problem, options=options)

# Run optimisation
results = optim.run()
print(results)
print(f"Initial volumetric power density: {-results.initial_cost:.2f} W.m-3")
print(f"Optimised volumetric power density: {-results.best_cost:.2f} W.m-3")

# Plot the cost landscape with optimisation path
results.plot_surface()

# Obtain the optimised pybamm.ParameterValues object for use with PyBaMM classes
optimised_values = results.parameter_values

# Plot comparison
sim_original = pybamm.Simulation(
    model, parameter_values=parameter_values, experiment=experiment
)
sol_original = sim_original.solve(initial_soc=1.0)
sim_optimised = pybamm.Simulation(
    model, parameter_values=optimised_values, experiment=experiment
)
sol_optimised = sim_optimised.solve(initial_soc=1.0)
pybop.plot.trajectories(
    x=[sol_original.t, sol_optimised.t],
    y=[sol_original["Voltage [V]"].data, sol_optimised["Voltage [V]"].data],
    trace_names=["Original", "Optimised"],
    xaxis_title="Time / s",
    yaxis_title="Voltage / V",
)
