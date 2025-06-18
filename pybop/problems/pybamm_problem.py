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
        use_last_cost_index: list[bool] = None,
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
        self._use_last_cost_index = use_last_cost_index

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

    def _compute_cost(self, solution: list[Solution]) -> list[float]:
        """
        Compute the cost function value from a solution.
        """
        costs = [
            sol[name].data[-1] if use_last else sol[name].data
            for use_last, name in zip(
                self._use_last_cost_index, self._cost_names, strict=False
            )
            for sol in solution
        ]
        # NOTE: The below is needed until pybamm issue: https://github.com/pybamm-team/PyBaMM/issues/5063
        # is closed. The below code sums across the time axis, replicating the processing that occurs
        # when output_variables is not used for the DiscreteTimeSum node.
        if costs[0].shape != self._params.current_value().shape:
            costs = np.sum(costs, axis=1).reshape(self._cost_weights.shape[0], -1)
        weighted_costs = np.dot(self._cost_weights, costs)
        if weighted_costs.size == 1:
            return weighted_costs[0]
        return np.dot(self._cost_weights, costs)

    def _add_prior_contribution(self, cost: float) -> float:
        """
        Add the prior contribution to the cost if using posterior.
        """
        if not self._use_posterior:
            return cost

        # Likelihoods and priors are negative by convention
        return cost - self._priors.logpdf(self._params.get_values())

    def _compute_cost_with_prior(self, solution: Solution) -> float:
        """
        Compute the cost function with optional prior contribution.
        """
        cost = self._compute_cost(solution)
        return self._add_prior_contribution(cost)

    def run(self) -> float:
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

    def run_with_sensitivities(self) -> tuple[float, np.ndarray]:
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

        # Below is a patch for output_variables until the
        # DiscreteTimeSum functionality is added.
        aggregated_sens = np.asarray(
            [
                np.sum(s[n].sensitivities["all"], axis=0)
                for n in self._cost_names
                for s in sol
            ]
        )
        weighted_sensitivity = np.sum(
            aggregated_sens * self._cost_weights[:, None], axis=0
        )

        # Add prior derivative contribution if using posterior
        if self._use_posterior:
            for param_idx, _param_name in enumerate(self._params.keys()):
                weighted_sensitivity[param_idx] -= prior_derivatives[param_idx]

        return cost, weighted_sensitivity

    @property
    def pipeline(self):
        return self._pipeline
