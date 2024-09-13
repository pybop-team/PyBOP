from typing import Union

import numpy as np

import pybop

"""
When fitting empirical models, the parameters we are able to identify
will be constrained from the data that's available. For example, it's
no good trying to fit an RC timescale of 0.1 s from data sampled at
1 Hz! Likewise, an RC timescale of 100 s cannot be meaningfully fitted
to just 10 s of data. To ensure the optimiser doesn't propose
excessively long or short timescales - beyond what can reasonably be
inferred from the data - it is common to apply nonlinear constraints
on the parameter space. This script fits an RC pair with the
constraint 0.5 <= R1 * C1 <= 1, to highlight a possible method for
applying constraints on the timescales.

An alternative approach is given i the ecm_trust-constr notebook,
which can lead to better results and higher optimisation efficiency
when good timescale guesses are available.
"""

# Import the ECM parameter set from JSON
# parameter_set = pybop.ParameterSet(
#     json_path="examples/scripts/parameters/initial_ecm_parameters.json"
# )
# parameter_set.import_parameters()


# Alternatively, define the initial parameter set with a dictionary
# Add definitions for R's, C's, and initial overpotentials for any additional RC elements
parameter_set = {
    "chemistry": "ecm",
    "Initial SoC": 0.75,
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
    tau_mins: Union[float, list[float]],
    tau_maxs: Union[float, list[float]],
    fitted_rc_pair_indices: Union[int, list[int]],
):
    """Returns a function to check parameters against given tau bounds.
    The resulting check_params function will be sent off to PyBOP; the
    rest of the code does some light checking of the constraints.

    Parameters
    ----------
    tau_mins: float or list[float]
        Lower bounds on timescale tau_i = Ri * Ci
    tau_maxs: float or list[float]
        Upper bounds on timescale tau_i = Ri * Ci
    fitted_rc_pair_indices: int or list[float]
        The index of each RC pair whose parameters are to be fitted.
        Eg. [1, 2] means fitting R1, R2, C1, C2. The timescale of RC
        pair fitted_rc_pair_indices[j] is constrained to be in the
        range tau_mins[j] <= R * C <= tau_maxs[j]

    Returns
    -------
    check_params
        Function to check the proposed parameter values match the
        requested constraints

    """

    # Ensure inputs are lists
    tau_mins = [tau_mins] if not isinstance(tau_mins, list) else tau_mins
    tau_maxs = [tau_maxs] if not isinstance(tau_maxs, list) else tau_maxs
    fitted_rc_pair_indices = (
        [fitted_rc_pair_indices]
        if not isinstance(fitted_rc_pair_indices, list)
        else fitted_rc_pair_indices
    )

    # Validate input lengths
    if len(tau_mins) != len(fitted_rc_pair_indices) or len(tau_maxs) != len(
        fitted_rc_pair_indices
    ):
        raise ValueError(
            "tau_mins and tau_maxs must have the same length as fitted_rc_pair_indices"
        )

    def check_params(
        inputs: dict[str, float] = None,
        parameter_set=None,
        allow_infeasible_solutions: bool = False,
    ) -> bool:
        """Checks if the given inputs are within the tau bounds."""
        # Allow simulation to run if inputs are None
        if inputs is None or inputs == {}:
            return True

        # Check every respective R*C against tau bounds
        for i, tau_min, tau_max in zip(fitted_rc_pair_indices, tau_mins, tau_maxs):
            tau = inputs[f"R{i} [Ohm]"] * inputs[f"C{i} [F]"]
            if not tau_min <= tau <= tau_max:
                return False
        return True

    return check_params


# Define the model
params = pybop.ParameterSet(params_dict=parameter_set)
model = pybop.empirical.Thevenin(
    parameter_set=params,
    check_params=get_parameter_checker(
        0, 1.0, 1
    ),  # Set the model up to automatically check parameters
    options={"number of rc elements": 2},
)

# Fitting parameters
parameters = pybop.Parameters(
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
)

sigma = 0.001
t_eval = np.arange(0, 600, 3)
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
cost = pybop.RootMeanSquaredError(problem)
optim = pybop.XNES(
    cost,
    sigma0=[1e-4, 1e-4, 100],  # Set parameter specific step size
    allow_infeasible_solutions=False,
    max_unchanged_iterations=30,
    max_iterations=125,
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
