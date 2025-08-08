import numpy as np

import pybop
from pybop.parameters.multivariate_parameters import MultivariateParameters


def BaseOptimiser_run_MonkeyPatch(self):
    results = []
    for i in range(self._multistart):
        if i >= 1:
            initial_values = self.problem.params.rvs(1)[0]
            self.problem.params.update(initial_values=initial_values)
            self._set_up_optimiser()
        results.append(self._run())

    result = results[0].__class__.combine(results)

    self.problem.params.update(values=result.x_best)

    if self._logger.verbose:
        print(result)


pybop.BaseOptimiser.run = BaseOptimiser_run_MonkeyPatch


class BayesianOptimisationResult(pybop.OptimisationResult):
    """
    Stores the result of a Bayesian optimisation or a Bayesian model
    selection.

    Attributes
    ----------
    problem: pybop.Problem
        The optimisation object used to generate the results.
    x : ndarray
        The solution of the optimisation (in model space).
    final_cost : float
        The cost associated with the solution x.
    n_iterations : int or dict
        Number of iterations performed by the optimiser. Since Bayesian
        optimisers tend to have layers of various optimisation
        algorithms, their iteration counts may be put individually.
    n_evaluations : int or dict
        Number of evaluations performed by the optimiser. Since Bayesian
        optimisers tend to have layers of various optimisation
        algorithms, their evaluation counts my be put individually.
    message : str
        The reason for stopping given by the optimiser.
    lower_bounds: ndarray
        The lower confidence parameter boundaries.
    upper_bounds: ndarray
        The upper confidence parameter boundaries.
    posterior : MultivariateParameters
        The probability distribution of the optimisation.
    maximum_a_posteriori : Inputs or ndarray
        Complementing the best observed value in `x`, this is the
        prediction for the best parameter value.
    log_evidence_mean : float
        The logarithm of the evidence of the parameterization. Higher
        values are better. May only be interpreted relative to a
        calibration case, e.g., a test-run with synthetic data.
    log_evidence_variance : float
        The logarithm of the variance in the calculation of the
        evidence. For reliable comparisons based on the evidence, should
        be at or below the scale of the evidence itself.
    """

    def __init__(
        self,
        problem: pybop.Problem,
        x: np.ndarray,
        final_cost: float,
        n_iterations: int | dict,
        n_evaluations: int | dict,
        time: float | dict,
        message: str | None = None,
        lower_bounds: np.ndarray | None = None,
        upper_bounds: np.ndarray | None = None,
        posterior: MultivariateParameters | None = None,
        maximum_a_posteriori: np.ndarray | None = None,
        log_evidence_mean: float | None = None,
        log_evidence_variance: float | None = None,
    ):
        super().__init__(
            problem=problem,
            x=x,
            final_cost=final_cost,
            n_iterations=n_iterations,
            n_evaluations=n_evaluations,
            time=time,
            message=message,
        )
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.posterior = posterior
        self.maximum_a_posteriori = maximum_a_posteriori
        self.log_evidence_mean = log_evidence_mean
        self.log_evidence_variance = log_evidence_variance

    @staticmethod
    def combine(
        results: list["BayesianOptimisationResult"],
    ) -> "BayesianOptimisationResult":
        """
        Combine multiple BayesianOptimisationResult objects into a single one.

        Parameters
        ----------
        results : list[BayesianOptimisationResult]
            List of BayesianOptimisationResult objects to combine.

        Returns
        -------
        BayesianOptimisationResult
            Combined BayesianOptimisationResult object.
        """
        return pybop.OptimisationResult.combine(results)
