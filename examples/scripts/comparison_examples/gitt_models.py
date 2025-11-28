import numpy as np
import pybamm
from scipy import stats

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
corrupt_values = solution["Voltage [V]"].data + np.random.normal(
    0, sigma, len(solution.t)
)
dataset = pybop.Dataset(
    {
        "Time [s]": solution.t,
        "Current function [A]": solution["Current [A]"].data,
        "Discharge capacity [A.h]": solution["Discharge capacity [A.h]"].data,
        "Voltage [V]": corrupt_values,
    }
)

for model in [pybop.lithium_ion.WeppnerHuggins(), pybop.lithium_ion.SPDiffusion()]:
    # GITT target parameter
    diffusion_parameter = pybop.ParameterDistribution(stats.norm(5000, 1000))
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
        grouped_parameter_values.update(
            {
                "Particle diffusion time scale [s]": diffusion_parameter,
                "Reference voltage [V]": pybop.ParameterInfo(
                    initial_value=grouped_parameter_values["Reference voltage [V]"],
                ),
            }
        )

    else:
        # Group parameter values
        grouped_parameter_values = (
            pybop.lithium_ion.SPDiffusion.create_grouped_parameters(parameter_values)
        )

        # Fitting parameters
        grouped_parameter_values.update(
            {
                "Particle diffusion time scale [s]": diffusion_parameter,
                "Series resistance [Ohm]": pybop.ParameterInfo(
                    initial_value=grouped_parameter_values["Series resistance [Ohm]"],
                ),
            }
        )

    # Build the problem
    gitt_dataset = (
        dataset.get_subset(pulse_index)
        if isinstance(model, pybop.lithium_ion.WeppnerHuggins)
        else dataset
    )
    simulator = pybop.pybamm.Simulator(
        model, parameter_values=grouped_parameter_values, protocol=gitt_dataset
    )
    cost = pybop.RootMeanSquaredError(gitt_dataset, weighting="domain")
    problem = pybop.Problem(simulator, cost)

    # Build the optimisation problem
    optim = pybop.SciPyMinimize(problem)

    # Run the optimisation problem
    result = optim.run()
    print(result)
    print(
        "Diffusion time [s]:", result.best_inputs["Particle diffusion time scale [s]"]
    )

    # Plot the timeseries output
    pybop.plot.problem(problem, inputs=result.best_inputs, title=model.name)

print(
    "Note the different optimised values for the particle diffusion time scale,"
    " which is a consequence of the differing model assumptions."
)
