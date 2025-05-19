from typing import Union

import numpy as np

from pybop import Parameters
from pybop._pybamm_pipeline import PybammPipeline
from pybop.problems.base_problem import Problem


class PybammProblem(Problem):
    """
    Defines a problem that uses a PyBaMM model as the simulation + cost function to evaluate
    """

    def __init__(
        self,
        pybamm_pipeline: PybammPipeline,
        pybop_params: Parameters = None,
        cost_names: list[str] = None,
        cost_weights: Union[list, np.array] = None,
    ):
        super().__init__(pybop_params=pybop_params)
        self._pipeline = pybamm_pipeline
        self._cost_names = cost_names
        self._cost_weights = cost_weights
        self._domain = "Time [s]"

    def set_params(self, p: np.ndarray) -> None:
        """
        Sets the parameters for the simulation and cost function.
        """
        self.check_and_store_params(p)

        # rebuild the pipeline (if needed)
        self._pipeline.rebuild(self._params.as_dict())

    def run(self) -> float:
        """
        Evaluates the underlying simulation and cost function using the
        parameters set in the previous call to `set_params`.
        """
        self.check_set_params_called()

        # run simulation
        sol = self._pipeline.solve()

        # extract and sum cost function values. These are assumed to all be scalar values
        return np.dot(self._cost_weights, [sol[n].data for n in self._cost_names])

    def run_with_sensitivities(
        self,
    ) -> tuple[float, np.ndarray]:
        """
        Evaluates the underlying simulation and cost function using the
        parameters set in the previous call to `set_params`.
        """

        self.check_set_params_called()

        # run simulation
        sol = self._pipeline.solve(calculate_sensitivities=True)

        # extract cost function values. These are assumed to all be scalar values
        cost = np.dot(self._cost_weights, [sol[n].data for n in self._cost_names])

        # sensitivities will all be 1D arrays of length n_params, sum over the different
        # cost functions to get the total sensitivity
        # Note: current pybamm version has a bug where the sensitivities are not
        # summed over the discrete time points, so we do that here
        # REMEMBER TO REMOVE THIS WHEN THE BUG IS FIXED!!!
        # https://github.com/pybamm-team/PyBaMM/pull/5008
        cost_sens = []
        for param_name in self._params.keys():
            aggregated_sens = np.asarray(
                [
                    np.sum(sol[cost_name].sensitivities[param_name], axis=0)
                    for cost_name in self._cost_names
                ]
            )

            # Remove unnecessary dimensions and apply weights
            weighted_sensitivity = np.dot(
                np.squeeze(aggregated_sens), self._cost_weights
            )

            cost_sens.append(weighted_sensitivity)

        return cost, np.array(cost_sens)
