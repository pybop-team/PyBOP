import pybamm
from pybamm import Parameter

import pybop
from pybop.pybamm.parameter_utils import set_formation_concentrations

# Define model
model = pybamm.lithium_ion.SPMe()

# Supplement the model with design variables
target_time = 1800  # must match the length of dis/charge in the experiment [s]
pybop.pybamm.add_variable_to_model(
    model, "Gravimetric power density [W.kg-1]", target_time=target_time
)
pybop.pybamm.add_variable_to_model(
    model, "Volumetric power density [W.m-3]", target_time=target_time
)

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

# Define an initial guess for the discharge rate
discharge_rate = 2 * parameter_values["Nominal cell capacity [A.h]"]

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
    ["Discharge at 1C for 30 minutes or until 2.5 V (5 seconds period)"]
)

# Generate problem
simulator = pybop.pybamm.Simulator(
    model,
    parameter_values,
    input_parameter_names=parameters.names,
    protocol=experiment,
    initial_state={"Initial SoC": 1.0},
)
problem = pybop.DesignProblem(
    simulator,
    parameters,
    output_variables=[
        "Voltage [V]",
        "Gravimetric power density [W.kg-1]",
        "Volumetric power density [W.m-3]",
    ],
)

# Generate multiple cost functions and combine them
cost1 = pybop.DesignCost(problem, target="Gravimetric power density [W.kg-1]")
cost2 = pybop.DesignCost(problem, target="Gravimetric power density [W.kg-1]")
cost = pybop.WeightedCost(cost1, cost2, weights=[1, 1e-3])

# Run the optimisation
options = pybop.PintsOptions(max_iterations=10)
optim = pybop.XNES(cost, options=options)
result = optim.run()
print(result)
print(f"Initial gravimetric power density: {cost1(result.x0):.2f} W.kg-1")
print(f"Optimised gravimetric power density: {cost1(result.x):.2f} W.kg-1")
print(f"Initial volumetric power density: {cost2(result.x0):.2f} W.m-3")
print(f"Optimised volumetric power density: {cost2(result.x):.2f} W.m-3")

# Plot the timeseries output
pybop.plot.problem(problem, problem_inputs=result.x, title="Optimised Comparison")

# Plot the optimisation result
pybop.plot.surface(optim)
