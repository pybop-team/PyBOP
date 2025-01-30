import numpy as np

import pybop
from pybop.models.lithium_ion.basic_SP_diffusion import (
    convert_physical_to_electrode_parameters,
)
from pybop.models.lithium_ion.weppner_huggins import convert_physical_to_gitt_parameters

# Define model
parameter_set = pybop.ParameterSet("Xu2019")
model = pybop.lithium_ion.SPMe(
    parameter_set=parameter_set, options={"working electrode": "positive"}
)

# Generate data
sigma = 1e-3
initial_state = {"Initial SoC": 0.9}
experiment = pybop.Experiment(
    [
        "Rest for 1 second",
        "Discharge at 1C for 10 minutes (10 second period)",
        "Rest for 20 minutes",
    ]
)
values = model.predict(initial_state=initial_state, experiment=experiment)
corrupt_values = values["Voltage [V]"].data + np.random.normal(
    0, sigma, len(values["Voltage [V]"].data)
)

# Form dataset and locate the pulse
dataset = pybop.Dataset(
    {
        "Time [s]": values["Time [s]"].data,
        "Current function [A]": values["Current [A]"].data,
        "Discharge capacity [A.h]": values["Discharge capacity [A.h]"].data,
        "Voltage [V]": corrupt_values,
    }
)

for model_type in [pybop.lithium_ion.WeppnerHuggins, pybop.lithium_ion.SPDiffusion]:
    # GITT target parameter
    diffusion_parameter = pybop.Parameter(
        "Particle diffusion time scale [s]",
        prior=pybop.Gaussian(5000, 1000),
    )

    if model_type == pybop.lithium_ion.WeppnerHuggins:
        # Define parameter set
        parameter_set = convert_physical_to_gitt_parameters(
            model._unprocessed_parameter_set, "positive"
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
            * (parameter_set["Theoretical electrode capacity [A.s]"] / 3600)
        )
        parameter_set.update(
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
                initial_value=parameter_set["Reference voltage [V]"],
            ),
        )

    else:
        # Define parameter set
        parameter_set = convert_physical_to_electrode_parameters(
            model._unprocessed_parameter_set, "positive"
        )

        # Fitting parameters
        parameters = pybop.Parameters(
            diffusion_parameter,
            pybop.Parameter(
                "Series resistance [Ohm]",
                initial_value=parameter_set["Series resistance [Ohm]"],
            ),
        )

    # Define the cost to optimise
    gitt_model = model_type(parameter_set=parameter_set)
    problem = pybop.FittingProblem(
        gitt_model,
        parameters,
        dataset.get_subset(pulse_index)
        if model_type == pybop.lithium_ion.WeppnerHuggins
        else dataset,
    )
    cost = pybop.RootMeanSquaredError(problem)

    # Build the optimisation problem
    optim = pybop.SciPyMinimize(cost=cost)

    # Run the optimisation problem
    results = optim.run()

    # Plot the timeseries output
    pybop.plot.quick(problem, problem_inputs=results.x, title="Optimised Comparison")

print(
    "Note the different optimised values for the particle diffusion time scale,"
    " which is a consequence of the differing model assumptions."
)
