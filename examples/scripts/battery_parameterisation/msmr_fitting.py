"""
Please use the `develop` branch of PyBOP to run this script
pip install git+https://github.com/pybop-team/PyBOP@develop

This script can be used to refine estimates for MSMR parameters
from OCV data, provided the initial estimates are fairly accurate.

As well as changing the initial guesses, it can help to vary the
"weighting" between SoC and dSoC/dV in the cost function.
"""

import numpy as np
from scipy import constants

import pybop

parameter_set = pybop.ParameterSet("Chen2020")

for cell_type in ["negative electrode", "positive electrode"]:
    # Generate some synthetic OCP data for testing
    if cell_type == "positive electrode":
        ocp_function = parameter_set["Positive electrode OCP [V]"]
        soc_min, soc_max = 0.266145163492257, 0.7  # 0.905926128940627
    else:
        ocp_function = parameter_set["Negative electrode OCP [V]"]
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
        domain="SoC",
    )

    # Define the MSMR model parameters using a set number of electrode reactions
    n_reactions = 4
    T = 298.15
    F_RT = constants.physical_constants["Faraday constant"][0] / (constants.R * T)
    if cell_type == "positive electrode":
        msmr_params = {
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
    else:
        msmr_params = {
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

    def sub(s: str, j: int):
        return s + "_" + str(j)

    class FunctionFitting(pybop.FittingProblem):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.parameter_values = msmr_params

            # Define the state of charge
            def soc(U):
                X = self.parameter_values["X_shift"]
                for j in range(n_reactions):
                    X += self.p("X", j) / (
                        1 + np.exp(F_RT * (U - self.p("U0", j)) / self.p("w", j))
                    )
                return X

            self.soc = soc

        def p(self, s: str, j: int):
            return self.parameter_values[sub(s, j)]

        def evaluate(self, inputs=None):
            inputs = self.parameters.verify(inputs)
            self.parameter_values.update(inputs)

            soc = []
            for voltage in self.domain_data:
                try:
                    soc.append(self.soc(voltage))
                except (ZeroDivisionError, RuntimeError, ValueError) as e:
                    if isinstance(e, ValueError) and str(e) not in self.exception:
                        raise  # raise the error if it doesn't match the expected list
                    soc.append(
                        -1
                    )  # add impossible stoichiometry to increase the error measure

            return {
                "SoC": np.asarray(soc),
                "Voltage [V]": self.domain_data,
                "Weighted dSoC/dV": weighting
                * np.gradient(np.asarray(soc), self.domain_data),
            }

    for i in [0, 1, 2]:
        # Define the optimisation parameters
        U_max, w_max = 6, 10
        if i == 0:
            parameters = pybop.Parameters(
                pybop.Parameter(
                    "X_shift", initial_value=msmr_params["X_shift"], bounds=[-0.5, 0.5]
                ),
            )
            for j in range(n_reactions):
                parameters.add(
                    pybop.Parameter(
                        sub("X", j),
                        initial_value=msmr_params[sub("X", j)],
                        bounds=[0, 1],
                    )
                )
                parameters.add(
                    pybop.Parameter(
                        sub("w", j),
                        initial_value=msmr_params[sub("w", j)],
                        bounds=[0, w_max],
                    )
                )
        elif i == 1:
            for j in range(n_reactions):
                parameters.add(
                    pybop.Parameter(
                        sub("U0", j),
                        initial_value=msmr_params[sub("U0", j)],
                        bounds=[
                            msmr_params[sub("U0", j)] - 0.01,
                            msmr_params[sub("U0", j)] + 0.01,
                        ],
                    )
                )

        # Create the fitting problem
        problem = FunctionFitting(
            model=None,
            parameters=parameters,
            dataset=ocp_dataset,
            domain="Voltage [V]",
            signal=["SoC", "Weighted dSoC/dV"],
        )

        # Optimise the fit between the model and the dataset
        cost = pybop.SumSquaredError(problem, weighting="domain")
        x0 = parameters.initial_value()
        print("Initial cost:", cost(x0))
        optim = pybop.SciPyMinimize(cost=cost, tol=1e-8)
        results = optim.run()
        print(results)
        parameters.update(initial_values=results.x)
        msmr_params.update(parameters.as_dict(results.x))

    # Verify the method through plotting
    initial_solution = problem.evaluate(inputs=x0)
    optimised_solution = problem.evaluate(inputs=results.x)
    fig = pybop.plot.trajectories(
        x=[ocp_dataset["SoC"], initial_solution["SoC"], optimised_solution["SoC"]],
        y=[
            ocp_dataset["Voltage [V]"],
            initial_solution["Voltage [V]"],
            optimised_solution["Voltage [V]"],
        ],
        trace_names=["Ground truth", "Initial MSMR fit", "Optimised MSMR fit"],
        xaxis_title="State of charge",
        yaxis_title="Voltage [V]",
    )
    fig = pybop.plot.trajectories(
        x=[
            ocp_dataset["Weighted dSoC/dV"],
            initial_solution["Weighted dSoC/dV"],
            optimised_solution["Weighted dSoC/dV"],
        ],
        y=[
            ocp_dataset["Voltage [V]"],
            initial_solution["Voltage [V]"],
            optimised_solution["Voltage [V]"],
        ],
        trace_names=["Ground truth", "Initial MSMR fit", "Optimised MSMR fit"],
        xaxis_title="Weighted dSoC/dV",
        yaxis_title="Voltage [V]",
    )

    # Scale the results so that the total occupancy equals one
    X_sum = 0
    for X_j in [sub("X", j) for j in range(n_reactions)]:
        X_sum += msmr_params[X_j]
    print("X_sum:", X_sum)
    msmr_params["X_shift"] = msmr_params["X_shift"] / X_sum
    for X_j in [sub("X", j) for j in range(n_reactions)]:
        msmr_params[X_j] = msmr_params[X_j] / X_sum
    print(msmr_params)
