import pybamm

import pybop

# Define model and parameter values
model_options = {"working electrode": "positive"}
model = pybamm.lithium_ion.SPMe(options=model_options)
parameter_values = pybamm.ParameterValues("Xu2019")
parameter_values.set_initial_state(0.9, options=model_options)

# Generate a synthetic dataset
sigma = 1e-3
experiment = pybamm.Experiment(
    [
        "Rest for 1 second",
        "Discharge at 1C for 10 minutes (10 second period)",
        "Rest for 20 minutes",
    ]
)
solution = pybamm.Simulation(
    model, parameter_values=parameter_values, experiment=experiment
).solve()
dataset = pybop.Dataset(
    {
        "Time [s]": solution.t,
        "Current [A]": solution["Current [A]"].data,
        "Discharge capacity [A.h]": solution["Discharge capacity [A.h]"].data,
        "Voltage [V]": pybop.add_noise(solution["Voltage [V]"].data, sigma),
    }
)

# Group the parameters
grouped_parameter_values = pybop.li_half_cell.SPDiffusion.create_grouped_parameters(
    parameter_values
)

# Fit the GITT pulse using the single particle diffusion model
gitt_fit = pybop.GITTPulseFit(parameter_values=grouped_parameter_values)
gitt_result = gitt_fit(gitt_pulse=dataset)

# Plot the timeseries output
pybop.plot.problem(
    gitt_fit.problem, inputs=gitt_result.best_inputs, title="Optimised Comparison"
)

# Plot the optimisation result
gitt_result.plot_convergence()
gitt_result.plot_parameters()
