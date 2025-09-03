import numpy as np

from pybop.analysis.sensitivity_analysis import sensitivity_analysis
from pybop.parameters.parameter import Inputs, Parameters


class Problem:
    """
    Defines a `run` method that takes candidate parameter sets and returns the corresponding
    values of the cost and sensitivities if needed.
    """

    def __init__(
        self, pybop_params: Parameters | None = None, is_posterior: bool = False
    ):
        if pybop_params is None:
            self._param_names = []
        self._params = pybop_params
        self._param_names = pybop_params.keys()
        self._n_params = len(pybop_params)
        self._has_sensitivities = False
        self.is_posterior = is_posterior

    def get_finite_initial_cost(self):
        """
        Compute the absolute initial cost, resampling the initial parameters if needed.
        """
        x0 = self._params.get_initial_values()
        cost0 = np.abs(self.run(x0))
        nsamples = 0
        while np.isinf(cost0) and nsamples < 10:
            x0 = self._params.sample_from_priors()
            if x0 is None:
                break

            cost0 = np.abs(self.run(x0))
            nsamples += 1
        if nsamples > 0:
            self._params.update(initial_values=x0)

        if np.isinf(cost0):
            raise ValueError("The initial parameter values return an infinite cost.")
        return cost0

    @property
    def params(self) -> Parameters:
        """
        Returns the parameters set for the simulation and cost function.
        """
        return self._params

    @property
    def param_names(self) -> list[str]:
        """
        Returns the names of the parameters set for the simulation and cost function.
        """
        return self._param_names

    @property
    def has_sensitivities(self) -> bool:
        return self._has_sensitivities

    def sensitivity_analysis(self, n_samples: int = 256) -> dict:
        """
        Computes the parameter sensitivities on the combined cost function using
        SOBOL analysis. See pybop.analysis.sensitivity_analysis for more details.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples for SOBOL sensitivity analysis, performs best as a
            power of 2, i.e. 128, 256, etc.
        """
        return sensitivity_analysis(problem=self, n_samples=n_samples)

    def run(self, values: np.ndarray) -> np.ndarray:
        """
        Evaluates the underlying simulation and cost function.

        Parameters
        ----------
        values : np.ndarray or list[np.ndarray]
            Either one candidate parameter set (a 1D numpy array), a list of candidate parameter sets
            or a 2D array of candidate parameter sets.

        Returns
        -------
        costs : np.ndarray
            A 1D array of either a single cost value or a set of cost values.
        """
        inputs = self._params.to_inputs(values)

        costs = self._compute_costs(inputs=inputs)

        # Add optional prior contribution
        if self.is_posterior:
            batch_values = np.asarray(
                [np.fromiter(x.values(), dtype=np.float64) for x in inputs]
            ).T  # note the required transpose
            log_prior = self._priors.logpdf(batch_values)  # Shape: (n_inputs,)
            return costs - log_prior

        return costs

    def run_with_sensitivities(
        self, values: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluates the underlying simulation and cost function, returns the cost and sensitivities.

        Parameters
        ----------
        values : np.ndarray or list[np.ndarray]
            Either one candidate parameter set (a 1D numpy array), a list of candidate parameter sets
            or a 2D array of candidate parameter sets.

        Returns
        -------
        costs : np.ndarray
            A 1D array of either a single cost value or a set of cost values.
        sensitivities : np.ndarray
            Either a 1D array of the gradients of the cost with respect to each parameter, or a
            2D array of sets of gradients with shape (number of candidates, number of parameters).
        """
        inputs = self._params.to_inputs(values)

        costs, sens = self._compute_costs_and_sensitivities(inputs=inputs)

        # Subtract optional prior contribution and derivatives from negative log-likelihood
        if self.is_posterior:
            batch_values = np.asarray(
                [np.fromiter(x.values(), dtype=np.float64) for x in inputs]
            ).T  # note the required transpose
            log_prior, log_prior_sens = self._priors.logpdfS1(batch_values)
            costs -= log_prior  # Shape: (n_inputs,)
            sens -= log_prior_sens  # Shape: (n_inputs, n_params)

        if np.asarray(values).ndim == 1:
            return costs, sens.reshape(-1)
        return costs, sens

    def _compute_costs(self, inputs: list[Inputs]) -> np.ndarray:
        """
        Evaluates the underlying simulation and cost function.

        Parameters
        ----------
        inputs : list[Inputs]
            A list of input dictionaries.

        Returns
        -------
        costs : np.ndarray
            A 1D array of cost values with length (number of candidates).
        """
        raise NotImplementedError

    def _compute_costs_and_sensitivities(
        self, inputs: list[Inputs]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluates the underlying simulation and cost function, returns the cost and sensitivities.

        Parameters
        ----------
        inputs : list[Inputs]
            A list of input dictionaries.

        Returns
        -------
        costs : np.ndarray
            A 1D array of cost values with length (number of candidates).
        sensitivities : np.ndarray
            A 2D array of sets of gradients with shape (number of candidates, number of parameters).
        """
        raise NotImplementedError
