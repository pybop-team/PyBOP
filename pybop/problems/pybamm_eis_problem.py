from collections.abc import Callable

import numpy as np

from pybop.parameters.parameter import Inputs, Parameters
from pybop.pipelines._pybamm_eis_pipeline import PybammEISPipeline
from pybop.problems.base_problem import Problem


class PybammEISProblem(Problem):
    """
    Defines a problem that uses a PyBaMM model as the simulation to evaluate
    the electrochemical impedance via electrochemical impedance spectroscopy (EIS).
    """

    def __init__(
        self,
        eis_pipeline: PybammEISPipeline,
        pybop_params: Parameters | None = None,
        cost_function: Callable = None,
        fitting_data: np.ndarray | None = None,
    ):
        super().__init__(pybop_params=pybop_params)
        self._pipeline = eis_pipeline
        self._cost_function = cost_function
        self._fitting_data = fitting_data

    def _compute_costs(self, inputs: list[Inputs]) -> np.ndarray:
        """
        Evaluates the underlying simulation and cost function.

        Returns
        -------
        cost : np.ndarray
            A 1D array of cost values of length `len(inputs)`.
        """
        costs = np.empty(len(inputs))

        for i, x in enumerate(inputs):
            residual = self._pipeline.solve(x) - self._fitting_data
            costs[i] = self._cost_function(residual)

        return costs

    def simulate(self, inputs: Inputs | list[Inputs]) -> np.ndarray:
        return self._pipeline.solve(inputs)

    @property
    def pipeline(self):
        return self._pipeline

    @property
    def fitting_data(self):
        return self._fitting_data

    @property
    def has_sensitivities(self):
        return False
