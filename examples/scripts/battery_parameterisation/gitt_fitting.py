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
        ("Discharge at 1C for 10 minutes (10 second period)", "Rest for 20 minutes")
        * 8,
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

# Determine the indices corresponding to each pulse in the dataset
nonzero_index = np.concatenate(
    (
        [-1],
        np.flatnonzero(dataset["Current function [A]"]),
        [len(dataset["Current function [A]"]) + 1],
    )
)
pulse_starts = np.extract(
    nonzero_index[1:] - nonzero_index[:-1] != 1,  # check if there is a gap
    nonzero_index[1:],  # return the index at the start of the pulse
)
pulse_index = []
for start, finish in zip(pulse_starts[:-1], pulse_starts[1:], strict=False):
    pulse_index.append(
        np.concatenate(
            ([start - 1], [i for i in nonzero_index if i >= start and i < finish])
        )
    )

# Group the parameters
grouped_parameter_values = pybop.lithium_ion.SPDiffusion.create_grouped_parameters(
    parameter_values
)

# Fit each pulse of the GITT measurement
gitt_fit = pybop.GITTFit(
    gitt_dataset=dataset,
    pulse_index=pulse_index,
    parameter_values=grouped_parameter_values,
)
gitt_parameter_data = gitt_fit()

# Plot the parameters
pybop.plot.dataset(gitt_parameter_data, signal=["Particle diffusion time scale [s]"])
pybop.plot.dataset(gitt_parameter_data, signal=["Series resistance [Ohm]"])

# Run the identified model
identified_model = pybop.lithium_ion.SPDiffusion(build=True)
grouped_parameter_values["Current function [A]"] = pybamm.Interpolant(
    dataset["Time [s]"], dataset["Current function [A]"], pybamm.t
)
fitted_values = pybamm.Simulation(
    identified_model, parameter_values=grouped_parameter_values
).solve(t_eval=dataset["Time [s]"], t_interp=dataset["Time [s]"])

# Return to the original model and update the diffusivity value
diffusivity = np.mean(
    parameter_values["Positive particle radius [m]"] ** 2
    / gitt_parameter_data["Particle diffusion time scale [s]"]
)
parameter_values.update({"Positive particle diffusivity [m2.s-1]": diffusivity})

# Compare the original, fitted and identified model predictions
sol = pybamm.Simulation(
    model, parameter_values=parameter_values, experiment=experiment
).solve()
pybop.plot.trajectories(
    sol.t,
    [
        dataset["Voltage [V]"],
        fitted_values["Voltage [V]"].data,
        sol["Voltage [V]"].data,
    ],
    trace_names=["Ground truth", "Fitted GITT Model", "Identified Model"],
    xaxis_title="Time / s",
    yaxis_title="Voltage / V",
)
