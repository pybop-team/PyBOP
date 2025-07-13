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
        self._cost_weights = cost_weights
        self._fitting_data = fitting_data
        self._compute_initial_cost_and_resample()

    def set_params(self, p: np.ndarray) -> None:
        """
        Sets the parameters for the simulation and cost function.
        """
        self.check_and_store_params(p)

        # rebuild the pipeline (if needed)
        self._pipeline.rebuild(self._params.to_dict())

    def run(self) -> float:
        """
        Evaluates the underlying simulation and cost function using the
        parameters set in the previous call to `set_params`.
        """
        self.check_set_params_called()

        # run simulation
        res = self._pipeline.solve() - self._fitting_data

        return np.dot(self._cost_weights, [cost(res) for cost in self._costs])

    def simulate(self, inputs: Inputs) -> np.ndarray:
        self._pipeline.rebuild(inputs)
        return self._pipeline.solve()

    @property
    def pipeline(self):
        return self._pipeline

    @property
    def fitting_data(self):
        return self._fitting_data
