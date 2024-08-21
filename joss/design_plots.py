
# A script to generate design optimisation plots for the JOSS paper.

import pybop

# Define parameter set and model
parameter_set = pybop.ParameterSet.pybamm("Chen2020", formation_concentrations=True)
model = pybop.lithium_ion.SPMe(parameter_set=parameter_set)

# Fitting parameters
parameters = pybop.Parameters(
    pybop.Parameter(
        "Positive electrode thickness [m]",
        initial_value=7.56e-05,
        prior=pybop.Gaussian(7.56e-05, 3e-05),
        bounds=[7e-05, 14e-05],
    ),
    pybop.Parameter(
        "Positive electrode active material volume fraction",
        initial_value=0.58,
        prior=pybop.Gaussian(0.58, 0.05),
        bounds=[0.3, 0.9],
    ),
)

# Define test protocol
experiment = pybop.Experiment(["Discharge at 1C until 2.5 V (5 seconds period)"])

# Generate problem, cost and optimiser classes
problem = pybop.DesignProblem(
    model,
    parameters,
    experiment,
    signal=["Voltage [V]", "Current [A]"],
    initial_state={"Initial SoC": 1.0},
    # update_capacity=True,
)
cost = pybop.GravimetricEnergyDensity(problem)
optim = pybop.XNES(
    cost,
    verbose=True,
    allow_infeasible_solutions=False,
    max_iterations=250,
    max_unchanged_iterations=25,
)

# Run optimisation
x, final_cost = optim.run()
print("Estimated parameters:", x)
print(f"Initial gravimetric energy density: {cost(optim.x0):.2f} Wh.kg-1")
print(f"Optimised gravimetric energy density: {cost(x):.2f} Wh.kg-1")

# Plot the timeseries output
pybop.quick_plot(problem, problem_inputs=x, title="Optimised Comparison")

# Plot the cost landscape with optimisation path
pybop.plot2d(optim, steps=15)
