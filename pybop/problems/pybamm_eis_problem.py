import numpy as np

from pybop import Parameters
from pybop._pybamm_eis_pipeline import PybammEISPipeline
from pybop.parameters.parameter import Inputs
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
        # self._compute_initial_cost_and_resample()

    def set_params(self, p: np.ndarray) -> None:
        """
        Sets the parameters for the simulation and cost function.
        """
        self.check_and_store_params(p)

    def run(self) -> np.ndarray:
        """
        Evaluates the underlying simulation and cost function using the
        parameters set in the previous call to `set_params`.

        Returns
        -------
        np.ndarray
            cost (np.ndarray): 1D weighted sum of cost variables.
        """
        inputs = self._params.to_pybamm_multiprocessing()
        cost_matrix = np.empty((len(self._costs), len(inputs)))

        for idx, param in enumerate(inputs):
            # Rebuild if required
            self._pipeline.rebuild(param)
            self._pipeline.initialise_eis_pipeline(param)

            # run simulation
            res = self._pipeline.solve() - self._fitting_data

            # Weighted cost w/ new axis to ensure the returned object is np.ndarray
            cost_matrix[:, idx] = [cost(res) for cost in self._costs]

        weighted_costs = self._cost_weights @ cost_matrix

        return weighted_costs

    def simulate(self, inputs: Inputs) -> np.ndarray:
        # self.set_params(p=np.asarray([v for v in inputs.values()]))
        for key, value in inputs.items():
            self._pipeline.pybamm_pipeline.parameter_values[key] = value
        self._pipeline.pybamm_pipeline.build()
        self._pipeline.initialise_eis_pipeline(inputs)
        return self._pipeline.solve()

    @property
    def pipeline(self):
        return self._pipeline

    @property
    def fitting_data(self):
        return self._fitting_data
