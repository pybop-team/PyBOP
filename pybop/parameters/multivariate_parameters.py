import numpy as np

from pybop import Parameters
from pybop.parameters.multivariate_distributions import BaseMultivariateDistribution


class MultivariateParameters(Parameters):
    """
    Represents a correlated set of uncertain parameters within the PyBOP
    framework.

    This class encapsulates the definition of each of its parameters,
    including their names, bounds, and margins to ensure the parameter
    stays within feasible limits during optimisation or sampling.

    Parameters
    ----------
    parameter_list : pybop.Parameter or Dict
    distribution : pybop.BaseMultivariateDistribution
        The joint multivariate distribution.
    """

    def __init__(self, *args, distribution=None):
        self.distribution = distribution
        super().__init__(args)
        for param in self._parameters.values():
            # Ensure that no individual distributions are mixed with the joint
            # one. They may have been used for setting boundaries.
            param._prior = None  # noqa: SLF001

    def get_margins(self) -> list:
        """
        Collects the margins of all parameters.

        Returns
        -------
            array-like
                A list of the margin attributes of each parameter.
        """
        return [param._margin for param in self._parameters.values()]  # noqa: SLF001

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
        x = np.asarray([self.transformation().to_search(m) for m in x.T]).T
        return self.distribution.pdf(x)

    def rvs(
        self, n_samples: int = 1, random_state=None, apply_transform: bool = False
    ) -> np.ndarray:
        """
        Draw random samples from the joint parameters distribution.

        The samples are constrained to be within each parameter's bounds,
        excluding a pre-defined margin at the boundaries.

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

        # Constrain samples to be within bounds.
        bounds = self.get_bounds(transformed=True)
        margins = self.get_margins()
        for i in range(len(samples)):
            offset = margins[i] * (bounds["upper"][i] - bounds["lower"][i])
            samples[i] = np.clip(
                samples[i], bounds["lower"][i] + offset, bounds["upper"][i] - offset
            )

        if apply_transform:
            samples = np.asarray(
                [self.transformation().to_model(s) for s in samples.T]
            ).T

        return samples

    def priors(self) -> BaseMultivariateDistribution:
        return self.distribution
