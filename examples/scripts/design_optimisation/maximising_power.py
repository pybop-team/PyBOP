import pybamm
from pybamm import Parameter

import pybop

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

# Define an initial guess for the discharge rate
discharge_rate = 2 * parameter_values["Nominal cell capacity [A.h]"]

# Fitting parameters
parameter_values.update(
    {
        "Positive electrode thickness [m]": pybop.Gaussian(
            7.56e-05,
            0.5e-05,
            bounds=[65e-06, 10e-05],
        ),
        "Nominal cell capacity [A.h]": pybop.Gaussian(  # controls the C-rate in the experiment
            discharge_rate,
            0.2,
            bounds=[0.8 * discharge_rate, 1.2 * discharge_rate],
        ),
    }
)

# Define test protocol
experiment = pybamm.Experiment(
    ["Discharge at 1C for 30 minutes or until 2.5 V (5 seconds period)"]
)

# Generate problem
simulator = pybop.pybamm.Simulator(
    model,
    parameter_values,
    protocol=experiment,
    initial_state={"Initial SoC": 1.0},
)
cost_1 = pybop.DesignCost(target="Gravimetric power density [W.kg-1]")
problem_1 = pybop.Problem(simulator, cost_1)
cost_2 = pybop.DesignCost(target="Volumetric power density [W.m-3]")
problem_2 = pybop.Problem(simulator, cost_2)
problem = pybop.MetaProblem(problem_1, problem_2, weights=[1, 1e-3])

# Run the optimisation
options = pybop.PintsOptions(max_iterations=10)
optim = pybop.XNES(problem, options=options)
result = optim.run()
print(result)
print(f"Initial gravimetric power density: {problem_1(result.x0):.2f} W.kg-1")
print(f"Optimised gravimetric power density: {problem_1(result.x):.2f} W.kg-1")
print(f"Initial volumetric power density: {problem_2(result.x0):.2f} W.m-3")
print(f"Optimised volumetric power density: {problem_2(result.x):.2f} W.m-3")

# Plot the optimisation result
result.plot_surface()

# Plot the timeseries output
problem_1.target = "Voltage [V]"
pybop.plot.problem(problem_1, problem_inputs=result.x, title="Optimised Comparison")
