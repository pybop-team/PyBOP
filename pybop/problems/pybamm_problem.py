import numpy as np
from pybamm import Solution

from pybop import JointLogPrior, Parameters
from pybop.pipelines._pybamm_pipeline import PybammPipeline
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
        cost_weights: list | np.ndarray = None,
        is_posterior: bool = False,
    ):
        super().__init__(pybop_params=pybop_params)
        self._pipeline = pybamm_pipeline
        self._cost_names = cost_names or []
        self._cost_weights = (
            np.asarray(cost_weights)
            if cost_weights is not None
            else np.ones(len(self._cost_names))
        )
        self._domain = "Time [s]"
        self.is_posterior = is_posterior
        self._has_sensitivities = (
            False if pybamm_pipeline.initial_state is not None else True
        )

        # Set up priors if we're using the posterior
        if self.is_posterior and pybop_params is not None:
            self._priors = JointLogPrior(*pybop_params.priors())
        else:
            self._priors = None

    def _compute_costs(self, values: np.ndarray | list[np.ndarray]) -> np.ndarray:
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
        inputs = self._params.to_inputs(values)

        sols = self._pipeline.solve(inputs=inputs)
        costs = self._get_pybamm_cost(sols)

        # Add optional prior contribution
        if self.is_posterior:
            batch_values = np.asarray(
                [np.fromiter(x.values(), dtype=np.floating) for x in inputs]
            ).T  # note the required transpose
            log_prior = self._priors.logpdf(batch_values)  # Shape: (n_inputs,)
            return costs - log_prior

        return costs

    def _compute_costs_and_sensitivities(
        self, values: np.ndarray | list[np.ndarray]
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
        inputs = self._params.to_inputs(values)

        sols = self._pipeline.solve(inputs=inputs, calculate_sensitivities=True)
        costs = self._get_pybamm_cost(sols)
        sens = self._get_pybamm_sensitivities(sols)

        # Subtract optional prior contribution and derivatives from negative log-likelihood
        if self.is_posterior:
            batch_values = np.asarray(
                [np.fromiter(x.values(), dtype=np.floating) for x in inputs]
            ).T  # note the required transpose
            out = self._priors.logpdfS1(batch_values)
            log_prior = out[0]  # Shape: (n_inputs,)
            log_prior_sens = out[1]  # Shape: (n_inputs, n_params)
            return costs - log_prior, sens - log_prior_sens

        return costs, sens

    def _get_pybamm_cost(self, solution: list[Solution]) -> np.ndarray:
        """Compute the cost function value from a list of solutions."""
        cost_matrix = np.empty((len(self._cost_names), len(solution)))

        # Extract each cost
        for i, name in enumerate(self._cost_names):
            cost_matrix[i, :] = [sol[name].data[0] for sol in solution]

        # Apply the weighting
        return self._cost_weights @ cost_matrix

    def _get_pybamm_sensitivities(self, solution: list[Solution]) -> np.ndarray:
        """Compute the cost function value and sensitivities from a list of solutions."""
        sens_matrix = np.empty((len(solution), self._n_params))

        # Extract each sensitivity and apply the weighting
        for i, s in enumerate(solution):
            weighted_sens = np.zeros(self._n_params)
            for n in self._cost_names:
                sens = np.asarray(s[n].sensitivities["all"])  # Shape: (1, n_params)
                weighted_sens += np.sum(
                    sens * self._cost_weights, axis=0
                )  # Shape: (n_params,)
            sens_matrix[i, :] = weighted_sens

        return sens_matrix

    @property
    def pipeline(self):
        return self._pipeline

    @property
    def cost_names(self):
        return self._cost_names
