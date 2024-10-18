import pybamm

import pybop

# The aim of this script is to show how to systematically update
# design parameters which depend on the optimisation parameters.

# Define parameter set and model
parameter_set = pybop.ParameterSet.pybamm("Chen2020", formation_concentrations=True)
parameter_set.update(
    {
        "Positive electrode porosity": 1
        - pybamm.Parameter("Positive electrode active material volume fraction")
    }
)
model = pybop.lithium_ion.SPMe(parameter_set=parameter_set)

# Fitting parameters
parameters = pybop.Parameters(
    pybop.Parameter(
        "Positive electrode thickness [m]",
        prior=pybop.Gaussian(7.56e-05, 0.1e-05),
        bounds=[65e-06, 10e-05],
    ),
    pybop.Parameter(
        "Positive electrode active material volume fraction",
        prior=pybop.Gaussian(0.6, 0.15),
        bounds=[0.1, 0.9],
    ),
)

# Define test protocol
experiment = pybop.Experiment(
    [
        "Discharge at 1C until 2.5 V (5 seconds period)",
        "Hold at 2.5 V for 30 minutes or until 10 mA (5 seconds period)",
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

# Define the cost
cost = pybop.GravimetricEnergyDensity(problem)

# Run optimisation
optim = pybop.XNES(
    cost, verbose=True, allow_infeasible_solutions=False, max_iterations=10
)
results = optim.run()
print("Estimated parameters:", results.x)
print(f"Initial gravimetric energy density: {cost(optim.x0):.2f} Wh.kg-1")
print(f"Optimised gravimetric energy density: {cost(results.x):.2f} Wh.kg-1")

# Plot the timeseries output
pybop.plot.quick(problem, problem_inputs=results.x, title="Optimised Comparison")

# Plot the cost landscape with optimisation path
pybop.plot.surface(optim)
