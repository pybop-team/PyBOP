import numpy as np
from pybamm import Solution

from pybop import JointLogPrior, Parameters
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
        cost_weights: list | np.ndarray = None,
        use_posterior: bool = False,
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
        self._use_posterior = use_posterior

        # Set up priors if we're using the posterior
        if self._use_posterior and pybop_params is not None:
            self._priors = JointLogPrior(*pybop_params.priors())
        else:
            self._priors = None

    def set_params(self, p: np.ndarray) -> None:
        """
        Sets the parameters for the simulation and cost function.
        """
        self.check_and_store_params(p)

        # rebuild the pipeline (if needed)
        self._pipeline.rebuild(self._params.to_dict())

    def _compute_cost(self, solution: list[Solution]) -> np.ndarray | np.number:
        """
        Compute the cost function value from a solution.
        """
        n_solutions = len(solution)
        n_costs = len(self._cost_names)
        cost_matrix = np.empty((n_costs, n_solutions))

        # Extract each cost
        for cost_idx, name in enumerate(self._cost_names):
            cost_matrix[cost_idx, :] = [sol[name].data[0] for sol in solution]

        # Apply cost weights
        weighted_costs = self._cost_weights @ cost_matrix
        if weighted_costs.ndim == 1 and weighted_costs.size == 1:
            return weighted_costs[0]
        elif weighted_costs.ndim == 2 and weighted_costs.shape[0] == 1:
            return weighted_costs.reshape(-1)

        return weighted_costs

    def _add_prior_contribution(self, cost: np.number | np.ndarray) -> np.number:
        """
        Add the prior contribution to the cost if using posterior.
        """
        if not self._use_posterior:
            return cost

        # Likelihoods and priors are negative by convention
        return cost - self._priors.logpdf(self._params.get_values())

    def _compute_cost_with_prior(self, solution: list[Solution]) -> np.number:
        """
        Compute the cost function with optional prior contribution.
        """
        cost = self._compute_cost(solution)
        return self._add_prior_contribution(cost)

    def run(self) -> np.ndarray:
        """
        Evaluates the underlying simulation and cost function using the
        parameters set in the previous call to `set_params`.

        Returns:
            The computed cost value
        """
        self.check_set_params_called()

        # Run simulation
        sol = self._pipeline.solve()

        # Compute cost with optional prior contribution
        return self._compute_cost_with_prior(sol)

    def run_with_sensitivities(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluates the simulation and cost function with parameter sensitivities
        using the parameters set in the previous call to `set_params`.

        Returns:
            Tuple of (cost_value, sensitivities)
        """
        self.check_set_params_called()
        prior_derivatives = np.zeros(len(self._params))

        # Compute prior contribution and derivatives if using posterior
        if self._use_posterior:
            log_prior, prior_derivatives = self._priors.logpdfS1(
                self._params.get_values()
            )

        # Solve with sensitivities, calculate cost
        sol = self._pipeline.solve(calculate_sensitivities=True)
        cost = self._compute_cost_with_prior(sol)

        # Shape: (n_solutions, n_params)
        total_weighted_sens = np.empty((len(sol), len(self._params)))
        for i, s in enumerate(sol):
            weighted_sens = np.zeros(len(self._params))
            for n in self._cost_names:
                sens = np.asarray(s[n].sensitivities["all"])  # Shape: (1, n_params)
                weighted_sens += np.sum(
                    sens * self._cost_weights, axis=0
                )  # Shape: (n_params,)
            total_weighted_sens[i, :] = weighted_sens

        # Add prior derivative contribution if using posterior
        if self._use_posterior:
            total_weighted_sens -= prior_derivatives

        if total_weighted_sens.ndim == 2 and total_weighted_sens.shape[0] == 1:
            return cost, total_weighted_sens.reshape(-1)

        return cost, total_weighted_sens

    @property
    def pipeline(self):
        return self._pipeline
