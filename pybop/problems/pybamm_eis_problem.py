import numpy as np

from pybop import Parameters
from pybop.parameters.parameter import Inputs
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
        costs: list | None = None,
        cost_weights: list | np.ndarray | None = None,
        fitting_data: np.ndarray | None = None,
    ):
        super().__init__(pybop_params=pybop_params)
        self._pipeline = eis_pipeline
        self._costs = costs
        self._fitting_data = fitting_data
        self._cost_weights = (
            np.asarray(cost_weights) if cost_weights is not None else None
        )

    def _compute_costs(self, values: np.ndarray | list[np.ndarray]) -> np.ndarray:
        """
        Evaluates the underlying simulation and cost function.

        Returns
        -------
        cost : np.ndarray
            A 1D array containing the weighted sum of cost variables for each proposal.
        """
        inputs = self._params.to_inputs(values)
        cost_matrix = np.empty((len(self._costs), len(inputs)))

        for i, x in enumerate(inputs):
            res = self._pipeline.solve(x) - self._fitting_data

            # Weighted cost w/ new axis to ensure the returned object is np.ndarray
            cost_matrix[:, i] = [cost(res) for cost in self._costs]

        return self._cost_weights @ cost_matrix

    def simulate(self, inputs: Inputs | list[Inputs]) -> np.ndarray:
        return self._pipeline.solve(inputs)

    @property
    def pipeline(self):
        return self._pipeline

    @property
    def fitting_data(self):
        return self._fitting_data
