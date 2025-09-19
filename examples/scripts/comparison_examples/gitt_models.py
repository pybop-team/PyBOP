import numpy as np
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
sol = pybamm.Simulation(
    model, parameter_values=parameter_values, experiment=experiment
).solve()
corrupt_values = sol["Voltage [V]"].data + np.random.normal(0, sigma, len(sol.t))
dataset = pybop.Dataset(
    {
        "Time [s]": sol.t,
        "Current function [A]": sol["Current [A]"].data,
        "Discharge capacity [A.h]": sol["Discharge capacity [A.h]"].data,
        "Voltage [V]": corrupt_values,
    }
)

for model in [pybop.lithium_ion.WeppnerHuggins(), pybop.lithium_ion.SPDiffusion()]:
    # GITT target parameter
    diffusion_parameter = pybop.Parameter(
        "Particle diffusion time scale [s]",
        prior=pybop.Gaussian(5000, 1000),
    )

    if isinstance(model, pybop.lithium_ion.WeppnerHuggins):
        # Group parameter values
        grouped_parameter_values = (
            pybop.lithium_ion.WeppnerHuggins.create_grouped_parameters(parameter_values)
        )

        # We can fit only the duration of the pulse
        pulse_index = np.flatnonzero(dataset["Current function [A]"])

        # We linearise the open-circuit voltage function
        ocp_derivative = (
            (dataset["Voltage [V]"][-1] - dataset["Voltage [V]"][0])
            / (
                dataset["Discharge capacity [A.h]"][-1]
                - dataset["Discharge capacity [A.h]"][0]
            )
            * (grouped_parameter_values["Theoretical electrode capacity [A.s]"] / 3600)
        )
        grouped_parameter_values.update(
            {
                "Reference voltage [V]": dataset["Voltage [V]"][pulse_index[0]],
                "Derivative of the OCP wrt stoichiometry [V]": ocp_derivative,
            },
        )

        # Fitting parameters
        parameters = pybop.Parameters(
            diffusion_parameter,
            pybop.Parameter(
                "Reference voltage [V]",
                initial_value=grouped_parameter_values["Reference voltage [V]"],
            ),
        )

    else:
        # Group parameter values
        grouped_parameter_values = (
            pybop.lithium_ion.SPDiffusion.create_grouped_parameters(parameter_values)
        )

        # Fitting parameters
        parameters = pybop.Parameters(
            diffusion_parameter,
            pybop.Parameter(
                "Series resistance [Ohm]",
                initial_value=grouped_parameter_values["Series resistance [Ohm]"],
            ),
        )

    # Define the model, problem and cost to optimise
    gitt_dataset = (
        dataset.get_subset(pulse_index)
        if isinstance(model, pybop.lithium_ion.WeppnerHuggins)
        else dataset
    )
    simulator = pybop.pybamm.Simulator(
        model,
        parameter_values=grouped_parameter_values,
        input_parameter_names=parameters.names,
        protocol=gitt_dataset,
    )
    problem = pybop.FittingProblem(simulator, parameters, gitt_dataset)
    cost = pybop.RootMeanSquaredError(problem, weighting="domain")

    # Build the optimisation problem
    optim = pybop.SciPyMinimize(cost)

    # Run the optimisation problem
    result = optim.run()
    print(result)
    print("Diffusion time [s]:", result.x[0])

    # Plot the timeseries output
    pybop.plot.problem(problem, problem_inputs=result.x, title=model.name)

print(
    "Note the different optimised values for the particle diffusion time scale,"
    " which is a consequence of the differing model assumptions."
)
