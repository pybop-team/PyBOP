import pybop
import pybamm
import pandas as pd
import numpy as np


def getdata(x0):
    # Define the "ground truth" model with the default parameter set
    model = pybamm.lithium_ion.SPM()
    params = model.default_parameter_values

    # Overwrite the uncertain parameters
    params.update(
        {
            "Negative electrode active material volume fraction": x0[0],
            "Positive electrode active material volume fraction": x0[1],
        }
    )

    # Define the experimental protocol
    experiment = pybamm.Experiment(
        [
            (
                "Discharge at 2C for 5 minutes (1 second period)",
                "Rest for 2 minutes (1 second period)",
                "Charge at 1C for 5 minutes (1 second period)",
                "Rest for 2 minutes (1 second period)",
            ),
        ]
        * 2
    )

    # Run a forward simulation
    sim = pybamm.Simulation(model, experiment=experiment, parameter_values=params)

    # Return the simulation results
    return sim.solve()


# Define the initial values of the uncertain parameters
x0 = np.array([0.55, 0.63])

# Generate observations
solution = getdata(x0)
observations = [
    pybop.Dataset("Time [s]", solution["Time [s]"].data),
    pybop.Dataset("Current function [A]", solution["Current [A]"].data),
    pybop.Dataset("Voltage [V]", solution["Terminal voltage [V]"].data),
]

# Define the model with the default parameter set
model = pybop.models.lithium_ion.SPM()
model.parameter_set = model.pybamm_model.default_parameter_values

# Initialise the fitting parameters
params = [
    pybop.Parameter(
        "Negative electrode active material volume fraction",
        prior=pybop.Gaussian(0.5, 0.05),
        bounds=[0.35, 0.75],
    ),
    pybop.Parameter(
        "Positive electrode active material volume fraction",
        prior=pybop.Gaussian(0.65, 0.05),
        bounds=[0.45, 0.85],
    ),
]

# Define the cost to optimise
cost = pybop.RMSE()
signal = "Voltage [V]"

# Select optimiser
optimiser = pybop.NLoptOptimize(x0=params)

# Build the optimisation problem
parameterisation = pybop.Optimisation(
    cost=cost,
    dataset=observations,
    signal=signal,
    model=model,
    optimiser=optimiser,
    fit_parameters=params,
)

# Run the optimisation problem
x, output, final_cost, num_evals = parameterisation.run()

print("Estimated parameters:", x)  # x = [0.54452026, 0.63064801]
print("Final cost:", final_cost)
