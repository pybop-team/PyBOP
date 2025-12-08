"""
Please use the `develop` branch of PyBOP to run this script
pip install git+https://github.com/pybop-team/PyBOP@develop

This script can be used to refine estimates for MSMR parameters
from OCV data, provided the initial estimates are fairly accurate.

As well as changing the initial guesses, it can help to vary the
"weighting" between SoC and dSoC/dV in the cost function.
"""

import numpy as np
import pybamm
from scipy import constants

import pybop

parameter_values = pybamm.ParameterValues("Chen2020")

for cell_type in ["negative electrode", "positive electrode"]:
    # Generate some synthetic OCP data for testing
    if cell_type == "positive electrode":
        ocp_function = parameter_values["Positive electrode OCP [V]"]
        soc_min, soc_max = 0.266145163492257, 0.7  # 0.905926128940627
    else:
        ocp_function = parameter_values["Negative electrode OCP [V]"]
        soc_min, soc_max = 0.0312962309919435, 0.7  # 0.901446800739041

    soc = np.linspace(soc_min, soc_max, 101)
    ocp = ocp_function(soc)
    weighting = 1e-3
    ocp_dataset = pybop.Dataset(
        {
            "SoC": soc,
            "Voltage [V]": ocp,
            "Weighted dSoC/dV": weighting * np.gradient(soc, ocp),
        },
        domain="Voltage [V]",
    )

    # Define the MSMR model parameters using a set number of electrode reactions
    T = 298.15
    msmr_params = {
        "n_reactions": 4,
        "T": T,
        "F_RT": constants.physical_constants["Faraday constant"][0] / (constants.R * T),
    }
    if cell_type == "positive electrode":
        msmr_params.update(
            {
                "X_shift": 0.340,
                "U0_0": 3.751,
                "X_0": 0.311,
                "w_0": 1.647,
                "U0_1": 4.002,
                "X_1": 0.382,
                "w_1": 3.011,
                "U0_2": 4.185,
                "X_2": 0.087,
                "w_2": 0.297,
                "U0_3": 4.283,
                "X_3": 0.219,
                "w_3": 4.361,
            }
        )
    else:
        msmr_params.update(
            {
                "X_shift": 0.020,
                "U0_0": 0.090,
                "X_0": 0.622,
                "w_0": 0.042,
                "U0_1": 0.134,
                "X_1": 0.183,
                "w_1": 0.108,
                "U0_2": 0.215,
                "X_2": 0.105,
                "w_2": 0.780,
                "U0_3": 0.393,
                "X_3": 0.090,
                "w_3": 5.360,
            }
        )

    def sub(s: str, j: int):
        return s + "_" + str(j)

    # Create the fitting problem
    class OCVCurve(pybop.BaseSimulator):
        def __init__(self, msmr_params, dataset, weighting):
            self.msmr_params = msmr_params.copy()
            # Unpack the uncertain parameters from the parameter values
            parameters = pybop.Parameters()
            for name, param in msmr_params.items():
                if isinstance(param, pybop.Parameter):
                    parameters.add(name, param)
            super().__init__(parameters=parameters)
            self.domain_data = dataset["Voltage [V]"]
            self.weighting = weighting

        def batch_solve(self, inputs, calculate_sensitivities: bool = False):
            solutions = []
            for x in inputs:
                sol = self.solve(x)
                solutions.append(sol)
            return solutions

        def solve(self, inputs=None):
            p = self.msmr_params.copy()
            p.update(inputs)

            sol = pybop.Solution()
            soc = []
            for U in self.domain_data:
                try:
                    X = p["X_shift"]
                    for j in range(p["n_reactions"]):
                        X += p[sub("X", j)] / (
                            1
                            + np.exp(p["F_RT"] * (U - p[sub("U0", j)]) / p[sub("w", j)])
                        )
                    soc.append(X)

                except (ZeroDivisionError, RuntimeError, ValueError) as e:
                    if isinstance(e, ValueError) and str(e) not in self.exception:
                        raise  # raise the error if it doesn't match the expected list
                    soc.append(
                        -1
                    )  # add impossible stoichiometry to increase the error measure

            sol.set_solution_variable("SoC", data=np.asarray(soc))
            sol.set_solution_variable("Voltage [V]", data=self.domain_data)
            sol.set_solution_variable(
                "Weighted dSoC/dV",
                data=self.weighting * np.gradient(np.asarray(soc), self.domain_data),
            )
            return sol

    for i in [0, 1, 2]:
        # Define the optimisation parameters
        U_max, w_max = 6, 10
        if i == 0:
            msmr_params.update(
                {
                    "X_shift": pybop.Parameter(
                        initial_value=msmr_params["X_shift"], bounds=[-0.5, 0.5]
                    )
                }
            )
            for j in range(msmr_params["n_reactions"]):
                msmr_params.update(
                    {
                        sub("X", j): pybop.Parameter(
                            initial_value=msmr_params[sub("X", j)], bounds=[0, 1]
                        ),
                        sub("w", j): pybop.Parameter(
                            initial_value=msmr_params[sub("w", j)], bounds=[0, w_max]
                        ),
                    }
                )
        elif i == 1:
            for j in range(msmr_params["n_reactions"]):
                msmr_params.update(
                    {
                        sub("U0", j): pybop.Parameter(
                            initial_value=msmr_params[sub("U0", j)],
                            bounds=[
                                msmr_params[sub("U0", j)] - 0.01,
                                msmr_params[sub("U0", j)] + 0.01,
                            ],
                        )
                    }
                )

        # Create the fitting problem
        simulator = OCVCurve(
            msmr_params=msmr_params, dataset=ocp_dataset, weighting=weighting
        )
        cost = pybop.SumSquaredError(
            ocp_dataset, target=["SoC", "Weighted dSoC/dV"], weighting="domain"
        )
        problem = pybop.Problem(simulator, cost)

        # Optimise the fit between the model and the dataset
        x0 = problem.parameters.get_initial_values()
        options = pybop.SciPyMinimizeOptions(tol=1e-8)
        optim = pybop.SciPyMinimize(problem, options=options)
        results = optim.run()
        print(results)
        problem.parameters.update(initial_values=results.x)

    # Update parameters with identified values
    msmr_params.update(problem.parameters.to_dict(results.x))

    # Verify the method through plotting
    initial_solution = simulator.solve(inputs=problem.parameters.to_dict(x0))
    optimised_solution = simulator.solve(inputs=problem.parameters.to_dict(results.x))
    fig = pybop.plot.trajectories(
        x=[
            ocp_dataset["SoC"],
            initial_solution["SoC"].data,
            optimised_solution["SoC"].data,
        ],
        y=[
            ocp_dataset["Voltage [V]"],
            initial_solution["Voltage [V]"].data,
            optimised_solution["Voltage [V]"].data,
        ],
        trace_names=["Ground truth", "Initial MSMR fit", "Optimised MSMR fit"],
        xaxis_title="State of charge",
        yaxis_title="Voltage [V]",
    )
    fig = pybop.plot.trajectories(
        x=[
            ocp_dataset["Weighted dSoC/dV"],
            initial_solution["Weighted dSoC/dV"].data,
            optimised_solution["Weighted dSoC/dV"].data,
        ],
        y=[
            ocp_dataset["Voltage [V]"],
            initial_solution["Voltage [V]"].data,
            optimised_solution["Voltage [V]"].data,
        ],
        trace_names=["Ground truth", "Initial MSMR fit", "Optimised MSMR fit"],
        xaxis_title="Weighted dSoC/dV",
        yaxis_title="Voltage [V]",
    )

    # Scale the results so that the total occupancy equals one
    X_sum = 0
    for X_j in [sub("X", j) for j in range(msmr_params["n_reactions"])]:
        X_sum += msmr_params[X_j]
    print("X_sum:", X_sum)
    msmr_params["X_shift"] = msmr_params["X_shift"] / X_sum
    for X_j in [sub("X", j) for j in range(msmr_params["n_reactions"])]:
        msmr_params[X_j] = msmr_params[X_j] / X_sum
    print(msmr_params)
