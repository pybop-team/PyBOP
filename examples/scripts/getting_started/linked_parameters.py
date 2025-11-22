import pybamm
from pybamm import Parameter

import pybop

"""
In this example, we introduce the functionality to link parameters in the underlying
PyBaMM model. Linking parameters can ensure correlated parameters are consistently
updated, ensuring that physical definitions are maintained. For this example, we link
the electrode porosity, active material volume fraction and binder fraction.
"""

# Define model
model = pybamm.lithium_ion.SPMe()
pybop.pybamm.add_variable_to_model(model, "Gravimetric energy density [Wh.kg-1]")

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
    },
    check_already_exists=False,
)

# Add and link parameters
parameter_values.update(
    {
        "Positive electrode binder fraction": 0.02,
        "Negative electrode binder fraction": 0.02,
    },
    check_already_exists=False,
)
parameter_values.update(
    {
        "Positive electrode porosity": (
            1.0
            - Parameter("Positive electrode active material volume fraction")
            - Parameter("Positive electrode binder fraction")
        ),
        "Negative electrode porosity": (
            1.0
            - Parameter("Negative electrode active material volume fraction")
            - Parameter("Negative electrode binder fraction")
        ),
    }
)

# Fitting parameters
parameter_values.update(
    {
        "Positive electrode thickness [m]": pybop.Parameter(
            prior=pybop.Gaussian(7.56e-05, 0.1e-05),
            bounds=[65e-06, 10e-05],
        ),
        "Positive electrode active material volume fraction": pybop.Parameter(
            prior=pybop.Gaussian(0.6, 0.15),
            bounds=[0.1, 0.9],
        ),
    }
)

# Define test protocol
experiment = pybamm.Experiment(
    [
        "Discharge at 1C until 2.5 V (10 seconds period)",
        "Hold at 2.5 V for 30 minutes or until 10 mA (10 seconds period)",
    ],
)

# Build the problem
simulator = pybop.pybamm.Simulator(
    model,
    parameter_values=parameter_values,
    protocol=experiment,
    initial_state={"Initial SoC": 1.0},
)
cost = pybop.DesignCost(target="Gravimetric energy density [Wh.kg-1]")
problem = pybop.Problem(simulator, cost)

# Set up the optimiser
options = pybop.PintsOptions(verbose=True, max_iterations=10)
optim = pybop.XNES(problem, options=options)

# Run the optimisation
result = optim.run()
print(f"Initial gravimetric energy density: {problem(result.x0):.2f} Wh.kg-1")
print(f"Optimised gravimetric energy density: {problem(result.x):.2f} Wh.kg-1")

# Plot the optimisation result
result.plot_surface()

# Plot the timeseries output
problem.target = "Voltage [V]"
pybop.plot.problem(problem, inputs=result.best_inputs, title="Optimised Comparison")
