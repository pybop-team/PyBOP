import numpy as np

from pybop import Parameters
from pybop.parameters.multivariate_distributions import BaseMultivariateDistribution


class MultivariateParameters(Parameters):
    """
    Represents a correlated set of uncertain parameters within the PyBOP
    framework.

    This class encapsulates the definition of each of its parameters,
    including their names and bounds to ensure the parameter
    stays within feasible limits during optimisation or sampling.

    Parameters
    ----------
    parameters : pybop.Parameters or dict
    distribution : pybop.BaseMultivariateDistribution
        The joint multivariate distribution.
    """

    def __init__(self, parameters: dict | Parameters, distribution=None):
        self.distribution = distribution
        super().__init__(parameters)
        for param in self._parameters.values():
            # Ensure that no individual distributions are mixed with the joint
            # one. They may have been used for setting boundaries.
            param._distribution = None  # noqa: SLF001

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the probability density function in parameter space.
        Just use the pdf method of the distribution directly for search space.

        Parameters
        ----------
        x : np.ndarray
            The parameter vector(s) to evaluate the pdf at.

        Returns
        -------
            np.ndarray
                An array of the pdf values at each parameter vector.
        """
        if len(x.shape) == 1:  # one-dimensional
            x = np.atleast_2d(x)
        x = np.asarray([self.transformation.to_search(m) for m in x])
        return self.distribution.pdf(x)

    def sample_from_distribution(
        self, n_samples: int = 1, random_state=None, apply_transform: bool = False
    ) -> np.ndarray:
        return self.rvs(n_samples, random_state, apply_transform)

    def rvs(
        self, n_samples: int = 1, random_state=None, apply_transform: bool = False
    ) -> np.ndarray:
        """
        Draw random samples from the joint parameters distribution.

        The samples are constrained to be within each parameter's bounds.

        Parameters
        ----------
        n_samples : int
            The number of samples to draw (default: 1).
        random_state : int, optional
            The random state seed for reproducibility (default: None).
        apply_transform : bool
            If True, the transformation is applied to the output
            (default: False).

        Returns
        -------
            array-like
                A matrix (i.e., a 2D array) of samples drawn from the
                joint distribution inside parameter boundaries.
        """
        samples = self.distribution.rvs(n_samples, random_state=random_state)
        if samples.ndim < 2:
            samples = np.atleast_2d(samples)

        if apply_transform:
            samples = np.asarray([self.transformation.to_model(s) for s in samples])

        return samples

    def distribution(self) -> BaseMultivariateDistribution:
        return self.distribution
