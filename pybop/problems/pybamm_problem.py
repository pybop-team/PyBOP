from typing import Union

import numpy as np

from pybop import Inputs, Parameters
from pybop._pybamm_pipeline import PybammPipeline
from pybop.problems.base_problem import Problem


class PybammProblem(Problem):
    """
    Defines a problem that uses a PyBaMM model as the simulation + cost function to evaluate
    """

    def __init__(
        self,
        pybamm_pipeline: list[PybammPipeline],
        pybop_params: Parameters = None,
        cost_names: list[str] = None,
        simulation_weights: Union[list, np.array] = None,
        cost_weights: Union[list, np.array] = None,
    ):
        super().__init__(pybop_params=pybop_params)
        self._pipeline = pybamm_pipeline
        self._cost_names = cost_names
        self._simulation_weights = simulation_weights
        self._cost_weights = cost_weights

    def set_params(self, p: np.ndarray) -> None:
        """
        Sets the parameters for the simulation and cost function.
        """
        self.check_and_store_params(p)

        # rebuild the pipeline (if needed)
        for pipe in self._pipeline:
            pipe.rebuild(self._params.as_dict())

    def run(self, inputs: Inputs = None) -> float:
        """
        Evaluates the underlying simulation and cost function using the
        parameters set in the previous call to `set_params`.
        """
        cost = []
        for pipe in self._pipeline:
            self.check_set_params_called()

            # run simulation
            sol = pipe.solve(inputs=self._params.as_dict())

            # extract and sum cost function values. These are assumed to all be scalar values
            # (not to self: test this is true in tests....)
            cost.append(
                np.dot(self._cost_weights, [sol[n].values[0] for n in self._cost_names])
            )
        return np.dot(self._simulation_weights, np.array(cost))

    def run_with_sensitivities(
        self,
    ) -> tuple[float, np.ndarray]:
        """
        Evaluates the underlying simulation and cost function using the
        parameters set in the previous call to `set_params`.
        """
        cost = []
        cost_sens = []

        for pipe in self._pipeline:
            self.check_set_params_called()

            # run simulation
            sol = pipe.solve(calculate_sensitivities=True)

            # extract cost function values. These are assumed to all be scalar values
            # (note to self: test this is true in tests....)
            cost.append(
                np.dot(self._cost_weights, [sol[n].values[0] for n in self._cost_names])
            )

            # sensitivities will all be 1D arrays of length n_params, sum over the different
            # cost functions to get the total sensitivity
            cost_sens.append(
                np.array(
                    [
                        np.dot(
                            self._cost_weights,
                            [
                                sol[cost_n].sensitivities[param_n]
                                for cost_n in self._cost_names
                            ],
                        )
                        for param_n in self._params.keys()
                    ]
                )
            )
        return np.dot(self._simulation_weights, np.array(cost)), np.dot(
            self._simulation_weights, np.array(cost_sens)
        )
