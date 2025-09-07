from collections.abc import Callable

import numpy as np

from pybop import JointLogPrior
from pybop.parameters.parameter import Inputs, Parameters
from pybop.pipelines._pybamm_pipeline import PybammPipeline
from pybop.problems.base_problem import Problem


class PybammProblem(Problem):
    """
    Defines a problem that uses a PyBaMM model as the simulation + cost function to evaluate.
    """

    def __init__(
        self,
        pybamm_pipeline: PybammPipeline,
        pybop_params: Parameters = None,
        cost_function: Callable = None,
        sensitivities: Callable | None = None,
        is_posterior: bool = False,
    ):
        super().__init__(pybop_params=pybop_params, is_posterior=is_posterior)
        self._pipeline = pybamm_pipeline
        self._get_cost = cost_function
        self._get_sensitivities = sensitivities
        self._domain = "Time [s]"

        # Set up priors if we're using the posterior
        if self.is_posterior and pybop_params is not None:
            self._priors = JointLogPrior(*pybop_params.priors())
        else:
            self._priors = None

    def _compute_costs(self, inputs: list[Inputs]) -> np.ndarray:
        """
        Evaluates the underlying simulation and cost function.

        The parameters can be set as singular proposals: np.array(N,)
        Or as an array of multiple proposals: np.array(M, N), where
        M is the number of proposals to solve.

        Returns
        -------
        cost : np.ndarray
            Weighted sum of cost variables for each proposal.
            The dimensionality is np.ndarray(M,) to match the number of proposals.
        """
        sols = self._pipeline.solve(inputs=inputs)

        return self._get_cost(sols)

    def _compute_costs_and_sensitivities(
        self, inputs: list[Inputs]
    ) -> tuple[np.ndarray | np.ndarray]:
        """
        Evaluates the simulation and cost function with parameter sensitivities.

        The parameters can be set as singular proposals: np.array(N,)
        Or as an array of multiple proposals: np.array(M,N), where
        M is the number of proposals to solve.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple containing the cost and parameter sensitivities:

            - cost ( np.ndarray(M,) ): Weighted sum of cost values for each proposal.
            - sensitivities ( np.ndarray(M, n_params) ): Weighted sum of parameter gradients for each proposal.
        """
        sols = self._pipeline.solve(inputs=inputs, calculate_sensitivities=True)

        return self._get_cost(sols), self._get_sensitivities(sols)

    @property
    def pipeline(self):
        return self._pipeline

    @property
    def has_sensitivities(self):
        return (
            False
            if self._pipeline.initial_state is not None
            or self._get_sensitivities is None
            else True
        )
