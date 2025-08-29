import numpy as np

import pybop

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
        ("Discharge at 1C for 10 minutes (10 second period)", "Rest for 20 minutes")
        * 8,
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

# Define parameter set
parameter_set = pybop.lithium_ion.SPDiffusion.apply_parameter_grouping(
    model.parameter_set, "positive"
)

# Fit each pulse of the GITT measurement
gitt_fit = pybop.GITTFit(
    gitt_dataset=dataset,
    pulse_index=pulse_index,
    parameter_set=parameter_set,
)
gitt_parameter_data = gitt_fit()

# Plot the parameters
pybop.plot.dataset(gitt_parameter_data, signal=["Particle diffusion time scale [s]"])
pybop.plot.dataset(gitt_parameter_data, signal=["Series resistance [Ohm]"])

# Run the identified model
model = pybop.lithium_ion.SPDiffusion(parameter_set=parameter_set, build=True)
model.set_current_function(dataset)
fitted_values = model.predict(t_eval=dataset["Time [s]"])

# Return to the original model
parameter_set = pybop.ParameterSet("Xu2019")

# Update the diffusivity value
diffusivity = np.mean(
    parameter_set["Positive particle radius [m]"] ** 2
    / gitt_parameter_data["Particle diffusion time scale [s]"]
)
parameter_set.update({"Positive particle diffusivity [m2.s-1]": diffusivity})

# Compare the original, fitted and identified model predictions
model = pybop.lithium_ion.SPMe(
    parameter_set=parameter_set, options={"working electrode": "positive"}
)
values = model.predict(initial_state=initial_state, experiment=experiment)
pybop.plot.trajectories(
    values["Time [s]"].data,
    [
        dataset["Voltage [V]"],
        fitted_values["Voltage [V]"].data,
        values["Voltage [V]"].data,
    ],
    trace_names=["Ground truth", "Fitted GITT Model", "Identified Model"],
    xaxis_title="Time / s",
    yaxis_title="Voltage / V",
)
