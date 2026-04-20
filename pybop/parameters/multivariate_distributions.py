import numpy as np
import scipy.integrate as integrate
import scipy.stats as stats
from scipy.linalg import sqrtm

from pybop.parameters.distributions import (
    BaseDistribution,
    Distribution,
    Gaussian,
    LogNormal,
    Uniform,
)
from pybop.transformation.base_transformation import Transformation
from pybop.transformation.transformations import (
    ComposedTransformation,
    IdentityTransformation,
    LogTransformation,
    ScaledTransformation,
)


class BaseMultivariateDistribution(Distribution):
    """
    A base class for defining multivariate parameter distributions.

    This class extends ``pybop.Distribution`` with methods that reduce the
    output of a multivariate distribution to its individual dimensions.

    Note that, unlike in ``pybop.Distribution``, distribution attributes
    are stored in a dictionary, as multivariate distributions in SciPy
    do not follow the loc/scale convention.

    Attributes
    ----------
    distribution : scipy.stats.rv_continuous
        The underlying continuous random variable distribution.
    properties : dict
        A dictionary with distribution keyword argument names as string
        keys and their values as float values.
    """

    def support(self) -> np.ndarray:
        """Return the support as a numpy array of dimensions (2, n_parameters)."""
        return self.distribution.support(**self.properties)

    def pdf(self, x):
        """Calculates the probability density function (PDF) of the distribution at x."""
        return self.distribution.pdf(x, **self.properties)

    def logpdf(self, x):
        """Calculates the logarithm of the probability density function of the distribution at x."""
        return self.distribution.logpdf(x, **self.properties)

    icdf = None
    """Multivariate distributions have no invertible CDF."""

    def icdf_marginal(self, q, position: int):
        raise NotImplementedError

    def cdf(self, x):
        return self.distribution.cdf(x, **self.properties)

    def cdf_marginal(self, x, position: int):
        """
        Takes the marginal cumulative distribution function (CDF) at x.

        Parameters
        ----------
        x : numpy.ndarray
            The point(s) at which to evaluate the CDF.
        position : int
            The dimension to which to reduce the CDF to.

        Returns
        -------
        float
            The marginal cumulative distribution function value at x.
        """
        return integrate.quad(self.marginal(position), -np.inf, x)[0]

    def rvs(self, size: int = 1, random_state: int | None = None):
        """Generates random variates from the distribution."""
        return self.distribution.rvs(
            **self.properties, size=size, random_state=random_state
        )

    def cov(self):
        """Get the covariance matrix."""
        return self.properties["cov"]

    def std(self):
        """Get the square root of the covariance matrix."""
        return sqrtm(self.cov())

    def get_transformed_distribution(
        self, transform: Transformation
    ) -> BaseDistribution | None:
        """Get the transformed distribution in the search space."""
        transform = (
            transform
            if isinstance(transform, ComposedTransformation)
            else ComposedTransformation([transform] * self._n_parameters)
        )
        self.check_consistent_transformation(transform)

        if isinstance(transform, IdentityTransformation) or all(
            isinstance(t, IdentityTransformation) for t in transform.transformations
        ):
            return self

        transformed_distribution = self._transform(transform)

        if transformed_distribution is None:
            transform_name = (
                transform.transformations[0].name
                if isinstance(transform, ComposedTransformation)
                else transform.name
            )
            raise TypeError(
                f"The transformation of a {self.name} distribution by a "
                f"{transform_name} is undefined or not yet implemented."
            )
        return transformed_distribution

    def check_consistent_transformation(self, transform: Transformation) -> None:
        """Raise an error if a composed transformation contains incompatible types of transformation."""
        for transformation_category in [ScaledTransformation, LogTransformation]:
            transformation_in_category: list[bool] = [
                isinstance(t, transformation_category)
                for t in transform.transformations
            ]
            if any(transformation_in_category) and not all(transformation_in_category):
                raise TypeError(
                    f"All transformations must be of a similar type. Received {[t.name for t in transform.transformations]}"
                )

    def marginal(self, position: int):
        """Return univariate marginal distribution of parameter with index 'position'."""
        raise NotImplementedError


class MultivariateNonparametric(BaseMultivariateDistribution):
    """
    Represents a "freeform" distribution, i.e., one that is defined from
    a random sampling and a kernel density estimate on that sampling.

    Parameters
    ----------
    samples : numpy.ndarray
        The random variates to base the distribution on.
    """

    def __init__(self, samples: np.ndarray):
        super().__init__(
            distribution=stats.gaussian_kde(samples), n_parameters=samples.shape[0]
        )

    def mean(self):
        # scipy.stats.gaussian_kde does not have the method "mean"
        # compute mean based on dataset
        return np.mean(self.distribution.dataset)

    def std(self):
        # scipy.stats.gaussian_kde does not have the method "std"
        # compute standard deviation based on dataset
        return np.std(self.distribution.dataset, ddof=1)

    def rvs(self, size: int = 1, random_state: int | None = None):
        return self.distribution.resample(size, random_state)

    def marginal(self, position: int):
        return MultivariateNonparametric(self.distribution.marginal(position).dataset)


class MultivariateUniform(BaseMultivariateDistribution):
    """
    Represents a multivariate uniform distribution.

    Parameters
    ----------
    bounds : numpy.ndarray
        The lower and upper bounds for the uniform distribution as an array
        with dimensions (2, n_parameters).
    """

    def __init__(self, bounds: np.ndarray):
        bounds = np.asarray(bounds)
        super().__init__(
            distribution=stats.uniform,
            properties={"loc": bounds[0, :], "scale": bounds[1, :] - bounds[0, :]},
            n_parameters=bounds.shape[1],
        )

    def marginal(self, position: int):
        # return univariate uniform distribution
        return Uniform(
            lower=self.properties["loc"][position],
            upper=self.properties["loc"][position] + self.properties["scale"][position],
        )

    def _transform(self, transform: Transformation) -> BaseDistribution | None:
        """Get the transformed distribution in the search space."""
        if all(isinstance(t, ScaledTransformation) for t in transform.transformations):
            lower = transform.to_search(self.properties["loc"])
            upper = transform.to_search(
                self.properties["loc"] + self.properties["scale"]
            )
            return MultivariateUniform(bounds=[lower, upper])
        return None


class MultivariateGaussian(BaseMultivariateDistribution):
    """
    Represents a multivariate Gaussian (normal) distribution with a
    given mean and covariance.

    Parameters
    ----------
    mean : numpy.ndarray
        The mean (µ) of the multivariate Gaussian distribution.
    covariance: numpy.ndarray
        The covariance matrix (Σ) of the multivariate Gaussian
        distribution. Note that what is called σ in 1D would be the
        square root of Σ here.
    """

    def __init__(self, mean: np.ndarray, covariance: np.ndarray):
        super().__init__(
            distribution=stats.multivariate_normal,
            properties={"mean": np.asarray(mean), "cov": np.asarray(covariance)},
            n_parameters=len(mean),
        )

    def support(self) -> np.ndarray:
        """Return the support as a numpy array of dimensions (2, n_parameters)."""
        return np.asarray([[-np.inf, np.inf]] * self._n_parameters).T

    def mean(self):
        return self.properties["mean"]

    def cov(self):
        return self.properties["cov"]

    def marginal(self, position: int):
        # Note that what is called σ in 1D is the square root of Σ here
        return Gaussian(
            mean=self.properties["mean"][position],
            sigma=np.sqrt(self.properties["cov"][position, position]),
        )

    def _transform(self, transform: Transformation) -> BaseDistribution | None:
        """Get the transformed distribution in the search space."""
        if all(isinstance(t, ScaledTransformation) for t in transform.transformations):
            mean = transform.to_search(self.properties["mean"])
            return MultivariateGaussian(
                mean=mean,
                covariance=transform.convert_covariance_matrix(
                    self.properties["cov"], mean
                ),
            )
        return None


class MultivariateLogNormal(BaseMultivariateDistribution):
    """
    Represents a multivariate log-normal distribution with a
    given mean and covariance of the normally distributed log(x).

    Parameters
    ----------
    mean_log_x : numpy.ndarray
        The mean (µ) of the multivariate Gaussian distribution of log(x).
    covariance_log_x: numpy.ndarray
        The covariance matrix (Σ) of the multivariate Gaussian
        distribution of log(x). Note that what is called σ in 1D would be the
        square root of Σ here.
    """

    def __init__(self, mean_log_x, covariance_log_x):
        super().__init__(distribution=None, n_parameters=len(mean_log_x))
        self.distribution_log_x = stats.multivariate_normal
        self.properties_log_x = {
            "mean": np.asarray(mean_log_x),
            "cov": np.asarray(covariance_log_x),
        }
        self.properties = self._get_properties()

    def _get_properties(self):
        """
        Method to compute the mean and covariance of x given the mean and
        covariance of log(x).
        """
        mean = np.zeros(self._n_parameters)
        for i in range(self._n_parameters):
            mean[i] = np.exp(
                self.properties_log_x["mean"][i]
                + 0.5 * self.properties_log_x["cov"][i, i]
            )
        covariance = np.zeros((self._n_parameters, self._n_parameters))
        for i in range(self._n_parameters):
            for j in range(self._n_parameters):
                covariance[i, j] = (
                    (np.exp(self.properties_log_x["cov"][i, j]) - 1) * mean[i] * mean[j]
                )

        return {"mean": mean, "cov": covariance}

    def support(self) -> np.ndarray:
        """Return the support as a numpy array of dimensions (2, n_parameters)."""
        return np.asarray([[0, np.inf]] * self._n_parameters).T

    def mean(self):
        return self.properties["mean"]

    def cov(self):
        return self.properties["cov"]

    def pdf(self, x):
        if np.any(x <= 0):
            return 0.0

        # change of variable
        log_x = np.log(x)

        # get pdf of normal distribution for transformed variable
        mvn_pdf = self.distribution_log_x.pdf(log_x, **self.properties_log_x)

        # determinant of the jacobian of the transformation
        jacobian = np.prod(x)

        return mvn_pdf / jacobian

    def logpdf(self, x):
        if np.any(x <= 0):
            return -np.inf

        # change of variable
        log_x = np.log(x)

        # use normal distribution to get logpdf of transformed variable
        mvn_logpdf = self.distribution_log_x.logpdf(log_x, **self.properties_log_x)

        # log of determinant of jacobian
        return mvn_logpdf - np.sum(log_x)

    def cdf(self, x):
        if np.any(x <= 0):
            return 0.0

        # change of variable
        log_x = np.log(x)

        # use normal distribution to get cdf of transformed variable
        return self.distribution_log_x.cdf(log_x, **self.properties_log_x)

    def rvs(self, size: int = 1, random_state: int | None = None):
        # use normal distribution of log(x) to compute rvs
        return np.exp(
            self.distribution_log_x.rvs(
                **self.properties_log_x, size=size, random_state=random_state
            )
        )

    def marginal(self, position: int):
        # determine marginal distribution based on properties of distribution of log(x)
        return LogNormal(
            mean_log_x=self.properties_log_x["mean"][position],
            sigma=np.sqrt(self.properties_log_x["cov"][position, position]),
        )

    def _transform(self, transform: Transformation) -> BaseDistribution | None:
        """Get the transformed distribution in the search space."""
        if all(isinstance(t, LogTransformation) for t in transform.transformations):
            return MultivariateGaussian(
                mean=self.properties_log_x["mean"],
                covariance=self.properties_log_x["cov"],
            )
        return None


class MarginalDistribution(Distribution):
    """
    Represents a univariate marginal distribution of a pybop.BaseMultivariateDistribution.
    Relies on the "marginal" method of the multivariate distribution.

    Sub-class of `pybop.Distribution` with the additional properties
    "parent_distribution" and "position"

    Parameters
    ----------
    parent_distribution: pybop.BaseMultivariateDistribution
        A multivariate distribution
    position: int
        position of the parameter for which to create the marginal distribution
    """

    def __init__(
        self, parent_distribution: BaseMultivariateDistribution, position: int
    ):
        # Get marginal distribution and initialise parent class
        distribution = parent_distribution.marginal(position)
        super().__init__(distribution=distribution)

        # add additional properties
        self._position = position
        self.parent_distribution = parent_distribution

    @property
    def position(self) -> int:
        return self._position

    def get_transformed_distribution(
        self, transform: Transformation
    ) -> BaseDistribution | None:
        """
        Get the transformed distribution in the search space, by first transforming the
        parent distribution and then fetching the marginal distribution.
        """
        transformed_parent_distribution = (
            self.parent_distribution.get_transformed_distribution(transform)
        )
        return MarginalDistribution(transformed_parent_distribution, self._position)

    def __repr__(self) -> str:
        return f"Marginal, position {self.position}, parent {self.parent_distribution.__repr__()}"
