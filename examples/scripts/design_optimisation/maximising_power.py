import pybamm
from pybamm import Parameter

import pybop

# Define parameter set and additional parameters needed for the cost function
parameter_set = pybop.ParameterSet("Chen2020", formation_concentrations=True)
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

# Define useful quantities
nominal_capacity = parameter_set["Nominal cell capacity [A.h]"]
target_c_rate = 2
discharge_rate = target_c_rate * nominal_capacity

# Fitting parameters
parameters = pybop.Parameters(
    pybop.Parameter(
        "Positive electrode thickness [m]",
        prior=pybop.Gaussian(7.56e-05, 0.5e-05),
        bounds=[65e-06, 10e-05],
    ),
    pybop.Parameter(
        "Nominal cell capacity [A.h]",  # controls the C-rate in the experiment
        prior=pybop.Gaussian(discharge_rate, 0.2),
        bounds=[0.8 * discharge_rate, 1.2 * discharge_rate],
    ),
)

# Define test protocol
experiment = pybamm.Experiment(
    ["Discharge at 1C for 30 minutes or until 2.5 V (5 seconds period)"],
)
signal = ["Voltage [V]", "Current [A]"]

# Generate problem
problem = pybop.DesignProblem(
    model,
    parameters,
    experiment,
    signal=signal,
    initial_state={"Initial SoC": 1.0},
)

# Generate multiple cost functions and combine them
cost1 = pybop.GravimetricPowerDensity(problem, target_time=3600 / target_c_rate)
cost2 = pybop.VolumetricPowerDensity(problem, target_time=3600 / target_c_rate)
cost = pybop.WeightedCost(cost1, cost2, weights=[1, 1e-3])

# Run optimisation
optim = pybop.XNES(
    cost, verbose=True, allow_infeasible_solutions=False, max_iterations=10
)
results = optim.run()
print(f"Initial gravimetric power density: {cost1(optim.x0):.2f} W.kg-1")
print(f"Optimised gravimetric power density: {cost1(results.x):.2f} W.kg-1")
print(f"Initial volumetric power density: {cost2(optim.x0):.2f} W.m-3")
print(f"Optimised volumetric power density: {cost2(results.x):.2f} W.m-3")
print(
    f"Optimised discharge rate: {results.x[-1]:.2f} A = {results.x[-1] / nominal_capacity:.2f} C"
)

# Plot the timeseries output
pybop.plot.problem(problem, problem_inputs=results.x, title="Optimised Comparison")

# Plot the cost landscape with optimisation path
pybop.plot.surface(optim)
