import pybop
import pybamm
import numpy as np

# Define model
original_parameters = pybamm.ParameterValues("Xu2019")
model = pybop.lithium_ion.SPM(
    parameter_set=original_parameters, options={"working electrode": "positive"}
)

# Generate data
sigma = 0.005
t_eval = np.arange(0, 150, 2)
values = model.predict(t_eval=t_eval)
corrupt_values = values["Voltage [V]"].data + np.random.normal(0, sigma, len(t_eval))

# Form dataset
dataset = pybop.Dataset(
    {
        "Time [s]": t_eval,
        "Current function [A]": values["Current [A]"].data,
        "Voltage [V]": corrupt_values,
    }
)

# Define parameter set
parameter_set = pybamm.ParameterValues(
    {
        "Reference OCP [V]": 4.1821,
        "Derivative of the OCP wrt stoichiometry [V]": -1.38636,
        "Current function [A]": original_parameters["Current function [A]"],
        "Number of electrodes connected in parallel to make a cell": original_parameters[
            "Number of electrodes connected in parallel to make a cell"
        ],
        "Electrode width [m]": original_parameters["Electrode width [m]"],
        "Electrode height [m]": original_parameters["Electrode height [m]"],
        "Positive electrode active material volume fraction": original_parameters[
            "Positive electrode active material volume fraction"
        ],
        "Positive electrode porosity": original_parameters[
            "Positive electrode porosity"
        ],
        "Positive particle radius [m]": original_parameters[
            "Positive particle radius [m]"
        ],
        "Positive electrode thickness [m]": original_parameters[
            "Positive electrode thickness [m]"
        ],
        "Positive electrode diffusivity [m2.s-1]": original_parameters[
            "Positive electrode diffusivity [m2.s-1]"
        ],
        "Maximum concentration in positive electrode [mol.m-3]": original_parameters[
            "Maximum concentration in positive electrode [mol.m-3]"
        ],
    }
)

# Define the cost to optimise
signal = ["Voltage [V]"]
problem = pybop.GITT(
    model="Weppner & Huggins", parameter_set=parameter_set, dataset=dataset
)
cost = pybop.RootMeanSquaredError(problem)

# Build the optimisation problem
optim = pybop.Optimisation(cost=cost, optimiser=pybop.PSO, verbose=True)

# Run the optimisation problem
x, final_cost = optim.run()
print("Estimated parameters:", x)

# Plot the timeseries output
pybop.quick_plot(x, cost, title="Optimised Comparison")

# Plot convergence
pybop.plot_convergence(optim)

# Plot the parameter traces
pybop.plot_parameters(optim)
