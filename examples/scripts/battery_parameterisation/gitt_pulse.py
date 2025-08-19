import matplotlib.pyplot as plt
import numpy as np
import pybamm

import pybop

# Define model
parameter_values = pybamm.ParameterValues("Xu2019")
model = pybamm.lithium_ion.SPMe(options={"working electrode": "positive"})

# Generate the synthetic dataset
sigma = 1e-3
initial_state = 0.85
experiment = pybamm.Experiment(
    [
        "Rest for 1 second (1 second period)",
        "Discharge at 1C for 10 minutes (10 second period)",
        "Rest for 20 minutes (10 second period)",
    ]
)
sim = pybamm.Simulation(
    model=model,
    parameter_values=parameter_values,
    experiment=experiment,
)
sol = sim.solve(initial_soc=initial_state)

dataset = pybop.Dataset(
    {
        "Time [s]": sol.t,
        "Voltage [V]": sol["Voltage [V]"].data + np.random.normal(0, sigma, len(sol.t)),
        "Discharge capacity [A.h]": sol["Discharge capacity [A.h]"].data,
        "Current function [A]": sol["Current [A]"].data,
    }
)

# Define parameter set
parameter_values = pybop.lithium_ion.SPDiffusion.create_grouped_parameters(
    parameter_values
)

# Fit the GITT pulse using the single particle diffusion model
gitt_fit = pybop.GITTPulseFit(parameter_values=parameter_values)
results = gitt_fit(gitt_pulse=dataset)


# Now we have a results object. The first thing we can
# do is obtain the fully identified pybamm.ParameterValues object
# These can then be used with normal Pybamm classes.
identified_parameter_values = results.parameter_values

sim = pybamm.Simulation(
    pybop.lithium_ion.SPDiffusion(),
    parameter_values=identified_parameter_values,
    experiment=experiment,
)
identified_sol = sim.solve(calc_esoh=False)

# Plot identified model vs dataset values
fig, ax = plt.subplots()
ax.plot(dataset["Time [s]"], dataset["Voltage [V]"], label="Target")
ax.plot(identified_sol.t, identified_sol["Voltage [V]"].data, label="Fit")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Voltage [V]")
ax.legend()
plt.show()

# Plot convergence
pybop.plot.convergence(gitt_fit.optim)

# Plot the parameter traces
pybop.plot.parameters(gitt_fit.optim)
