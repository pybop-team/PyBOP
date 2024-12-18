# A script to generate design optimisation plots for the JOSS paper.

import time

import numpy as np
import pybamm
from pybamm import Parameter

import pybop
from pybop.plot import PlotlyManager

go = PlotlyManager().go
np.random.seed(8)

# Choose which plots to show and save
create_plot = {}
create_plot["gravimetric"] = True  # takes longer
create_plot["prediction"] = True


# Define parameter set and model
parameter_set = pybop.ParameterSet.pybamm("Chen2020", formation_concentrations=True)
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
        "Positive electrode porosity": 1.0
        - Parameter("Positive electrode active material volume fraction"),
    },
    check_already_exists=False,
)
model = pybop.lithium_ion.SPMe(
    parameter_set=parameter_set, solver=pybamm.IDAKLUSolver(atol=1e-6, rtol=1e-6)
)

# Fitting parameters
parameters = pybop.Parameters(
    pybop.Parameter(
        "Positive electrode thickness [m]",
        initial_value=8.88e-05,
        prior=pybop.Gaussian(7.56e-05, 3e-05),
        bounds=[50e-06, 120e-06],
        transformation=pybop.UnitHyperCube(lower=50e-6, upper=120e-6),
    ),
    pybop.Parameter(
        "Positive electrode active material volume fraction",
        initial_value=0.42,
        prior=pybop.Gaussian(0.58, 0.1),
        bounds=[0.3, 0.825],
        transformation=pybop.UnitHyperCube(lower=0.3, upper=0.825),
    ),
)

# Define test protocol
experiment = pybop.Experiment(["Discharge at 1C until 2.5 V (10 second period)"])

# Generate problem, cost and optimiser classes
problem = pybop.DesignProblem(
    model,
    parameters,
    experiment,
    signal=["Voltage [V]", "Current [A]"],
    initial_state={"Initial SoC": 1.0},
    update_capacity=True,
)
parameter_set.update(
    {
        "Positive electrode thickness [m]": parameters[
            "Positive electrode thickness [m]"
        ].initial_value
    }
)
parameter_set.update(
    {
        "Positive electrode active material volume fraction": parameters[
            "Positive electrode active material volume fraction"
        ].initial_value
    }
)
initial_capacity = problem.model.approximate_capacity(parameter_set)
problem.model.parameter_set.update({"Nominal cell capacity [A.h]": initial_capacity})

cost = pybop.GravimetricEnergyDensity(problem)
optim = pybop.NelderMead(
    cost,
    verbose=True,
    parallel=True,
    allow_infeasible_solutions=False,
    max_iterations=250,
    max_unchanged_iterations=35,
    sigma0=0.1,
)

# Run optimisation
result = optim.run()
print("Estimated parameters:", result.x)
print(f"Initial gravimetric energy density: {cost(result.x0):.2f} Wh.kg-1")
print(f"Optimised gravimetric energy density: {cost(result.x):.2f} Wh.kg-1")

if create_plot["gravimetric"]:
    # Plot the cost landscape with optimisation path
    gravimetric_fig = pybop.plot.contour(
        optim,
        steps=60,
        title=None,
        xaxis=dict(titlefont_size=14, tickfont_size=14),
        yaxis=dict(titlefont_size=14, tickfont_size=14),
    )
    gravimetric_fig.write_image("joss/figures/design_gravimetric.pdf")
    time.sleep(3)
    gravimetric_fig.write_image("joss/figures/design_gravimetric.pdf")

problem.update_capacity = False
problem.model.parameter_set.update({"Nominal cell capacity [A.h]": initial_capacity})
if create_plot["prediction"]:
    # Plot the timeseries output
    figs = pybop.plot.quick(
        problem,
        problem_inputs=result.x,
        title=None,
        width=576,
        height=576,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(titlefont_size=14, tickfont_size=14),
        yaxis=dict(titlefont_size=14, tickfont_size=14),
        show=False,
    )
    prediction_fig = figs[0]
    prediction_fig.data[0].name = f"Initial result: {cost(result.x0):.2f} Wh.kg-1"
    prediction_fig.data[1].name = f"Optimised result: {cost(result.x):.2f} Wh.kg-1"
    prediction_fig.show()
    prediction_fig.write_image("joss/figures/design_prediction.pdf")
