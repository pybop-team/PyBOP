import numpy as np
from multivariate_priors import BaseMultivariatePrior

from pybop import Parameters


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
    prior : pybop.BaseMultivariatePrior
        The joint multivariate prior.
    """

    def __init__(self, *args, prior=None):
        self.prior = prior
        super().__init__(*args)
        for param in self.param.values():
            # Ensure that no individual priors are mixed with the joint
            # one. They may have been used for setting boundaries.
            param.prior = None

    def get_margins(self) -> list:
        """
        Collects the margins of all parameters.

        Returns
        -------
            array-like
                A list of the margin attributes of each parameter.
        """
        return [param.margin for param in self.param.values()]

    def rvs(
        self, n_samples: int = 1, random_state=None, apply_transform: bool = False
    ) -> np.ndarray:
        """
        Draw random samples from the joint parameters prior distribution.

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
                joint prior distribution inside parameter boundaries.
        """
        samples = self.prior.rvs(n_samples, random_state=random_state)

        # Constrain samples to be within bounds.
        bounds = self.get_bounds(apply_transform=False)
        margins = self.get_margins()
        for i in len(samples):
            offset = margins[i] * (bounds["upper"][i] - bounds["lower"][i])
            samples[i] = np.clip(
                samples[i], bounds["lower"][i] + offset, bounds["upper"][i] + offset
            )

        transformations = self.get_transformations()
        if apply_transform:
            for i in len(samples):
                if transformations[i]:
                    samples[i] = np.asarray(
                        [transformations[i].to_search(x) for x in samples[i]]
                    )

        return samples

    def get_sigma0(self, apply_transform: bool = False) -> list:
        # if apply_transform:
        #     raise NotImplementedError("Correlations may not sensibly transform.")
        try:
            return self.prior.sigma
        except NotImplementedError:
            return

    def priors(self) -> BaseMultivariatePrior:
        return self.prior
