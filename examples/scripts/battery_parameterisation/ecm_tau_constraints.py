import numpy as np
import pybamm

import pybop

"""
When fitting empirical models, the parameters we are able to identify will be constrained
from the data that's available. For example, it's no good trying to fit an RC timescale of
0.1 s from data sampled at 1 Hz! Likewise, an RC timescale of 100 s cannot be meaningfully
fitted to just 10 s of data. To ensure the optimiser doesn't propose excessively long or
short timescales - beyond what can reasonably be inferred from the data - it is common to
apply nonlinear constraints on the parameter space. This script fits an RC pair with the
constraint 0.5 <= R1 * C1 <= 1, to highlight a possible method for applying constraints on
the timescales.

An alternative approach is given i the ecm_trust-constr notebook, which can lead to better
results and higher optimisation efficiency when good timescale guesses are available.
"""


def get_parameter_checker(
    tau_mins: float | list[float],
    tau_maxs: float | list[float],
    fitted_rc_pair_indices: int | list[int],
):
    """
    Returns a function to check parameters against given tau bounds.
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
        parameter_values=None,
        allow_infeasible_solutions: bool = False,
    ) -> bool:
        """Checks if the given inputs are within the tau bounds."""
        # Allow simulation to run if inputs are None
        if inputs is None or inputs == {}:
            return True

        # Check every respective R*C against tau bounds
        for i, tau_min, tau_max in zip(
            fitted_rc_pair_indices, tau_mins, tau_maxs, strict=False
        ):
            tau = inputs[f"R{i} [Ohm]"] * inputs[f"C{i} [F]"]
            if not tau_min <= tau <= tau_max:
                return False
        return True

    return check_params


# Define the model
model = pybamm.equivalent_circuit.Thevenin(
    check_params=get_parameter_checker(
        0, 3.0, 1
    ),  # Set the model up to automatically check parameters
    options={"number of rc elements": 2},
)

# Define the initial parameter values
parameter_values = pybamm.ParameterValues("ECM_Example")
parameter_values.update(
    {
        "Initial SoC": 0.75,
        "Cell capacity [A.h]": 5,
        "Nominal cell capacity [A.h]": 5,
        "Current function [A]": 5,
        "Upper voltage cut-off [V]": 4.2,
        "Lower voltage cut-off [V]": 3.0,
        "Open-circuit voltage [V]": pybamm.empirical.Thevenin().default_parameter_values[
            "Open-circuit voltage [V]"
        ],
        "R0 [Ohm]": 0.001,
        "Element-1 initial overpotential [V]": 0,
        "R1 [Ohm]": 0.0002,
        "C1 [F]": 10000,
    }
)
# Add definitions for R's, C's, and initial overpotentials for any additional RC elements
parameter_values.update(
    {
        "Element-2 initial overpotential [V]": 0,
        "R2 [Ohm]": 0.0003,
        "C2 [F]": 5000,
    },
    check_already_exists=False,
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

results = optim.run()

# Plot convergence
results.plot_convergence()

# Plot the parameter traces
results.plot_parameters()

# Compare the fit to the data
pybop.plot.validation(results.x, problem=problem, dataset=dataset)
