# A script to generate design optimisation plots for the JOSS paper.


import numpy as np
import pybamm
from pybamm import Parameter

import pybop
from pybop.plot.backends import PlotlyManager

go = PlotlyManager().go
pybop.plot.use_backend("plotly")
np.random.seed(8)
axis_font_size = 24
tick_font_size = 20

# Choose which plots to show and save
create_plot = {}
create_plot["gravimetric"] = True  # takes longer
create_plot["prediction"] = True

# Define model and parameter values
model = pybamm.lithium_ion.SPMe()
pybop.pybamm.add_variable_to_model(model, "Gravimetric energy density [W.h.kg-1]")
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
        "Positive electrode porosity": 1.0
        - Parameter("Positive electrode active material volume fraction"),
        "Cell mass [kg]": pybop.pybamm.cell_mass(),
    },
    check_already_exists=False,
)

# Fitting parameters
parameter_values.update(
    {
        "Positive electrode thickness [m]": pybop.Parameter(
            initial_value=8.88e-05,
            distribution=pybop.Gaussian(
                7.56e-05, 3e-05, truncated_at=[50e-06, 120e-06]
            ),
            transformation=pybop.UnitHyperCube(lower=50e-6, upper=120e-6),
        ),
        "Positive electrode active material volume fraction": pybop.Parameter(
            initial_value=0.42,
            distribution=pybop.Gaussian(0.58, 0.1, truncated_at=[0.3, 0.825]),
            transformation=pybop.UnitHyperCube(lower=0.3, upper=0.825),
        ),
    }
)

# Define test protocol
experiment = pybamm.Experiment(["Discharge at 1C until 2.5 V (10 second period)"])
Q = parameter_values["Nominal cell capacity [A.h]"]
print(f"The 1C rate is {Q} A.")

# Generate problem
simulator = pybop.pybamm.Simulator(
    model,
    parameter_values=parameter_values,
    protocol=experiment,
    solver=pybamm.IDAKLUSolver(atol=1e-6, rtol=1e-6),
)
cost = pybop.DesignCost(target="Gravimetric energy density [W.h.kg-1]")
problem = pybop.Problem(simulator, cost)

# Set up the optimiser
options = pybop.PintsOptions(max_iterations=250, max_unchanged_iterations=45)
optim = pybop.NelderMead(problem, options=options)

# Run optimisation
result = optim.run()
print(result)
print("Estimated parameters:", result.x)
print(f"Initial gravimetric energy density: {problem(result.x0):.1f} W.h.kg-1")
print(f"Optimised gravimetric energy density: {problem(result.x):.1f} W.h.kg-1")
initial_energy_density = problem(result.x0)
optimised_energy_density = problem(result.x)

if create_plot["gravimetric"]:
    # Plot the cost landscape with optimisation path
    gravimetric_fig = pybop.plot.contour(result, steps=25, show=False)
    gravimetric_fig.update_layout(
        xaxis=dict(
            title=dict(
                text="Positive electrode thickness / m", font_size=axis_font_size
            ),
            tickfont_size=tick_font_size,
            exponentformat="power",
        ),
        yaxis=dict(title_font_size=axis_font_size, tickfont_size=tick_font_size),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font_size=tick_font_size,
        ),
        coloraxis_colorbar=dict(tickfont_size=tick_font_size),
        margin=dict(t=50),
        title=None,
    )
    gravimetric_fig.write_image("figures/individual/design_gravimetric.pdf")


if create_plot["prediction"]:
    # Plot the timeseries output
    problem.set_target("Voltage [V]")
    prediction_fig = pybop.plot.problem(
        problem,
        inputs=result.best_inputs,
        title=None,
        show=False,
    )
    prediction_fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font_size=tick_font_size - 2,
        ),
        xaxis=dict(
            showline=True,
            linewidth=1,
            linecolor="black",
            mirror=True,
            title_font_size=axis_font_size,
            tickfont_size=tick_font_size,
        ),
        yaxis=dict(
            showline=True,
            linewidth=1,
            linecolor="black",
            mirror=True,
            title_font_size=axis_font_size,
            tickfont_size=tick_font_size,
        ),
        margin=dict(t=60, b=84, r=50, l=15),
    )

    prediction_fig.data[1].update(line=dict(color="#00CC97"))
    prediction_fig.data[
        0
    ].name = f"Initial: {initial_energy_density:.1f} W&#8239;h&#8239;kg<sup>-1</sup>"
    prediction_fig.data[
        1
    ].name = (
        f"Optimised: {optimised_energy_density:.1f} W&#8239;h&#8239;kg<sup>-1</sup>"
    )
    prediction_fig.show()
    prediction_fig.write_image("figures/individual/design_prediction.pdf")
