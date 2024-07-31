import numpy as np

import pybop

# Import the ECM parameter set from JSON
# parameter_set = pybop.ParameterSet(
#     json_path="examples/scripts/parameters/initial_ecm_parameters.json"
# )
# parameter_set.import_parameters()


# Alternatively, define the initial parameter set with a dictionary
# Add definitions for R's, C's, and initial overpotentials for any additional RC elements
parameter_set = {
    "chemistry": "ecm",
    "Initial SoC": 0.5,
    "Initial temperature [K]": 25 + 273.15,
    "Cell capacity [A.h]": 5,
    "Nominal cell capacity [A.h]": 5,
    "Ambient temperature [K]": 25 + 273.15,
    "Current function [A]": 5,
    "Upper voltage cut-off [V]": 4.2,
    "Lower voltage cut-off [V]": 3.0,
    "Cell thermal mass [J/K]": 1000,
    "Cell-jig heat transfer coefficient [W/K]": 10,
    "Jig thermal mass [J/K]": 500,
    "Jig-air heat transfer coefficient [W/K]": 10,
    "Open-circuit voltage [V]": pybop.empirical.Thevenin().default_parameter_values[
        "Open-circuit voltage [V]"
    ],
    "R0 [Ohm]": 0.001,
    "Element-1 initial overpotential [V]": 0,
    "Element-2 initial overpotential [V]": 0,
    "R1 [Ohm]": 0.0002,
    "R2 [Ohm]": 0.0003,
    "C1 [F]": 10000,
    "C2 [F]": 5000,
    "Entropic change [V/K]": 0.0004,
}


def get_parameter_checker(
    tau_mins: float | list[float],
    tau_maxs: float | list[float],
    fitted_rc_pair_indices: int | list[int],
):
    if not hasattr(tau_mins, "len"):
        tau_mins = [tau_mins]
    if not hasattr(tau_maxs, "len"):
        tau_maxs = [tau_maxs]
    if not hasattr(fitted_rc_pair_indices, "len"):
        fitted_rc_pair_indices = [fitted_rc_pair_indices]

    if len(tau_mins) != len(fitted_rc_pair_indices):
        raise ValueError("tau_mins must have the same length as fitted_rc_pair_indices")
    if len(tau_maxs) != len(fitted_rc_pair_indices):
        raise ValueError("tau_maxs must have the same length as fitted_rc_pair_indices")

    def check_params(
        inputs: dict[str, float] = None,
        allow_infeasible_solutions: bool = False,
    ) -> bool:
        # Check every respective R*C <= tau_bound
        if inputs is None:
            # Simulating the model will result in this being called with
            # inputs=None; must return true to allow the simulation to run
            return True

        for i, tau_min, tau_max in zip(fitted_rc_pair_indices, tau_mins, tau_maxs):
            tau = inputs[f"R{i} [Ohm]"] * inputs[f"C{i} [F]"]
            if tau < tau_min:
                return False
            if tau > tau_max:
                return False
        return True

    return check_params


# Define the model
params = pybop.ParameterSet(params_dict=parameter_set)
model = pybop.empirical.Thevenin(
    parameter_set=params,
    check_params=get_parameter_checker(0, 0.5, 1),
    options={"number of rc elements": 2},
)

# Fitting parameters
parameters = [
    pybop.Parameter(
        "R0 [Ohm]",
        prior=pybop.Gaussian(0.0002, 0.0001),
        bounds=[1e-4, 1e-2],
    ),
    pybop.Parameter(
        "R1 [Ohm]",
        prior=pybop.Gaussian(0.0001, 0.0001),
        bounds=[1e-5, 1e-2],
    ),
    pybop.Parameter(
        "C1 [F]",
        prior=pybop.Gaussian(10000, 2500),
        bounds=[2500, 5e4],
    ),
]

sigma = 0.001
t_eval = np.arange(0, 900, 3)
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

# Generate problem, cost function, and optimisation class
problem = pybop.FittingProblem(model, parameters, dataset)
cost = pybop.SumSquaredError(problem)
optim = pybop.XNES(
    cost,
    allow_infeasible_solutions=False,
)

x, final_cost = optim.run()
print("Estimated parameters:", x)


# Plot the time series
pybop.plot_dataset(dataset)

# Plot the timeseries output
pybop.quick_plot(problem, problem_inputs=x, title="Optimised Comparison")

# Plot convergence
pybop.plot_convergence(optim)

# Plot the parameter traces
pybop.plot_parameters(optim)
