from typing import Union

import numpy as np

from pybop import Parameters
from pybop._pybamm_eis_pipeline import PybammEISPipeline
from pybop.problems.base_problem import Problem


class PybammEISProblem(Problem):
    """
    Defines a problem that uses a PyBaMM model as the simulation + cost function to evaluate
    the electrochemical impedance via electrochemical impedance spectroscopy (EIS).
    """

    def __init__(
        self,
        pybamm_pipeline: PybammEISPipeline,
        pybop_params: Parameters = None,
        costs: list = None,
        cost_weights: Union[list, np.array] = None,
        fitting_data: list = None,
    ):
        super().__init__(pybop_params=pybop_params)
        self._pipeline = pybamm_pipeline
        self._costs = costs
        self._cost_weights = cost_weights
        self._fitting_data = fitting_data

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
        self._pipeline.initialise_eis_pipeline()
        res = self._pipeline.solve() - self._fitting_data

        # extract and sum cost function values. These are assumed to all be scalar values
        # (note to self: test this is true in tests....)
        return np.dot(self._cost_weights, [cost(res) for cost in self._costs])
