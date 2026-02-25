import numpy as np
import scipy.integrate as integrate
import scipy.stats as stats
from scipy.linalg import sqrtm

from pybop.parameters.distributions import Distribution
from pybop.transformation.base_transformation import Transformation
from pybop.transformation.transformations import (
    ComposedTransformation,
    IdentityTransformation,
    LogTransformation,
    ScaledTransformation,
    UnitHyperCube,
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

    def pdf(self, x):
        return self.distribution.pdf(x, **self.properties)

    def logpdf(self, x):
        return self.distribution.logpdf(x, **self.properties)

    icdf = None
    """Multivariate distributions have no invertible CDF."""

    def icdf_marginal(self, q, dim):
        raise NotImplementedError

    def cdf(self, x):
        return self.distribution.cdf(x, **self.properties)

    def cdf_marginal(self, x, dim):
        """
        Takes the marginal cumulative distribution function (CDF) at x
        in dim.

        Parameters
        ----------
        x : numpy.ndarray
            The point(s) at which to evaluate the CDF.
        dim : int
            The dimension to which to reduce the CDF to.

        Returns
        -------
        float
            The marginal cumulative distribution function value at x in
            dim.
        """
        return integrate.quad(self.pdf_marginal, -np.inf, x)[0]

    def rvs(self, size=1, random_state=None):
        return self.distribution.rvs(
            **self.properties, size=size, random_state=random_state
        )

    def __repr__(self):
        return f"{self.__class__.__name__}, properties: {self.properties}"

    @property
    def mean(self):
        raise NotImplementedError

    @property
    def sigma(self):
        raise NotImplementedError

    @property
    def compatible_transformations(self):
        raise NotImplementedError

    def marginal(self, position):
        """Return univariate marginal distribution of parameter with index 'position'"""
        raise NotImplementedError

    def transformed_properties(self, transformation):
        """Return distribution properties in search space"""
        raise NotImplementedError


class MultivariateNonparametric(BaseMultivariateDistribution):
    """
    Represents a "freeform" distribution, i.e., one that is defined from
    a random sampling and a kernel density estimate on that sampling.

    This class provides methods to calculate the probability density
    function (pdf), the logarithm of the pdf, and to generate random
    variates (rvs) from the distribution.

    Parameters
    ----------
    samples : numpy.ndarray
        The random variates to base the distribution on.
    """

    def __init__(self, samples, random_state=None):
        super().__init__(distribution=stats.gaussian_kde(samples))
        self.name = "MultivariateNonparametric"
        self.properties = {}
        self._n_parameters = samples.shape[0]

    @property
    def compatible_transformations(self):
        return Transformation

    def pdf(self, x):
        return self.distribution.pdf(x)

    def logpdf(self, x):
        return self.distribution.logpdf(x)

    def rvs(self, size=1, random_state=None):
        return self.distribution.resample(size, random_state)

    def marginal(self, position):
        return self.distribution.marginal(position)

    def transformed_properties(self, transformation):
        """Return distribution properties in search space"""

        # properties is an empty dictionary and hence independent of transformation
        return self.properties


class MultivariateUniform(BaseMultivariateDistribution):
    """
    Represents a multivariate uniform distribution.

    This class provides methods to calculate the probability density
    fuction (pdf), the logarithm of the pdf, and to generate random
    variates (rvs) from the distribution.

    Parameters
    ----------
    bounds : numpy.ndarray
        The lower and upper bounds for the uniform distribution.
    """

    def __init__(self, bounds, random_state=None):
        super().__init__(
            distribution=stats.uniform(
                loc=bounds[0, :], scale=bounds[1, :] - bounds[0, :]
            )
        )
        self.name = "MultivariateUniform"
        self.properties = {"bounds": bounds}
        self._n_parameters = bounds.shape[1]

    @property
    def compatible_transformations(self):
        return (
            IdentityTransformation,
            ScaledTransformation,
            UnitHyperCube,
        )

    def marginal(self, position):
        # return univariate unfiform distribution
        return stats.uniform(
            loc=self.properties["bounds"][0, position],
            scale=self.properties["bounds"][1, position]
            - self.properties["bounds"][0, position],
        )


class MultivariateGaussian(BaseMultivariateDistribution):
    """
    Represents a multivariate Gaussian (normal) distribution with a
    given mean and covariance.

    This class provides methods to calculate the probability density
    function (pdf), the logarithm of the pdf, and to generate random
    variates (rvs) from the distribution.

    Parameters
    ----------
    mean : numpy.ndarray
        The mean (µ) of the multivariate Gaussian distribution.
    covariance: numpy.ndarray
        The covariance matrix (Σ) of the multivariate Gaussian
        distribution. Note that what is called σ in 1D would be the
        square root of Σ here.
    bounds : numpy.ndarray
        Lower and upper bounds (2nd dim) of the parameters (1st dim).
    """

    def __init__(self, mean=None, covariance=None, bounds=None, random_state=None):
        super().__init__(distribution=stats.multivariate_normal)
        self.name = "MultivariateGaussian"
        self.properties = {"mean": np.asarray(mean), "cov": np.asarray(covariance)}
        self.sigma2 = covariance
        self._n_parameters = len(mean)

    @property
    def compatible_transformations(self):
        return (IdentityTransformation, ScaledTransformation, UnitHyperCube)

    @property
    def mean(self):
        return self.properties["mean"]

    @property
    def sigma(self):
        return sqrtm(self.properties["cov"])

    def marginal(self, position):
        # Note that what is called σ in 1D is the square root of Σ here
        return stats.norm(
            loc=self.properties["mean"][position],
            scale=np.sqrt(self.properties["cov"][position, position]),
        )

    def transformed_properties(self, transformation):
        """Return distribution properties in search space"""
        if isinstance(transformation, self.compatible_transformations) or (
            isinstance(transformation, ComposedTransformation)
            and all(
                isinstance(trans, self.compatible_transformations)
                for trans in transformation.transformations
            )
        ):
            mean = transformation.to_search(self.properties["mean"])
            covariance = transformation.convert_covariance_matrix(
                self.properties["cov"], mean
            )

            return {"mean": mean, "cov": covariance}

        else:
            raise TypeError(
                "The transformation provided is not compatible with pybop.MultivariateGaussian. "
                "Only pybop.IdentityTransformation, pybop.ScaledTransformation and pybop.UnitHypercube are allowed."
            )


class MultivariateLogNormal(BaseMultivariateDistribution):
    """
    Represents a multivariate log-normal distribution with a
    given mean and covariance of the normally distributed log(x).

    This class provides methods to calculate the probability density
    function (pdf), the logarithm of the pdf, and to generate random
    variates (rvs) from the distribution.

    Parameters
    ----------
    mean_log_x : numpy.ndarray
        The mean (µ) of the multivariate Gaussian distribution of log(x).
    covariance_log_x: numpy.ndarray
        The covariance matrix (Σ) of the multivariate Gaussian
        distribution of log(x). Note that what is called σ in 1D would be the
        square root of Σ here.
    """

    def __init__(self, mean_log_x=None, covariance_log_x=None):
        self.name = "MultivariateLogNormal"
        self.distribution_log_x = stats.multivariate_normal
        self.properties_log_x = {
            "mean": np.asarray(mean_log_x),
            "cov": np.asarray(covariance_log_x),
        }
        self._n_parameters = len(mean_log_x)
        mean, covariance = self._get_covariance_and_mean()
        self.sigma2 = covariance
        self.properties = {"mean": mean, "cov": covariance}

    @property
    def compatible_transformations(self):
        return (LogTransformation, IdentityTransformation)

    def _get_covariance_and_mean(self):
        """Method to compute the covariance and mean in the model space based on the
        covariance and mean of log(x) (i.e. in the search space)"""
        covariance = np.zeros((self._n_parameters, self._n_parameters))
        mean = np.zeros(self._n_parameters)
        for i in range(self._n_parameters):
            mean[i] = np.exp(
                self.properties_log_x["mean"][i]
                + 0.5 * self.properties_log_x["cov"][i, i]
            )

        for i in range(self._n_parameters):
            for j in range(self._n_parameters):
                covariance[i, j] = (
                    (np.exp(self.properties_log_x["cov"][i, j]) - 1) * mean[i] * mean[j]
                )
        return mean, covariance

    def pdf(self, x):
        if np.any(x <= 0):
            return 0.0

        # change of variable
        log_x = np.log(x)

        # get pdf of normal distribution for transformed variable
        mvn_pdf = self.distribution_log_x.pdf(log_x, **self.properties_log_x)

        # determinant of the jacobion of the transformation
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

    def rvs(self, size=1, random_state=None):
        # use normal distribution of log(x) to compute rvs
        return np.exp(
            self.distribution_log_x.rvs(
                **self.properties_log_x, size=size, random_state=random_state
            )
        )

    def marginal(self, position):
        # determine marginal distribution based on properties of distribution of log(x)
        return stats.lognorm(
            scale=np.exp(self.properties_log_x["mean"][position]),
            s=np.sqrt(self.properties_log_x["cov"][position, position]),
        )

    def transformed_properties(self, transformation):
        """Return distribution properties in search space"""

        # only allow Indentity- and LogTransformation and use properties or properties_log_x to return correct properties
        if isinstance(transformation, LogTransformation) or (
            isinstance(transformation, ComposedTransformation)
            and all(
                isinstance(trans, LogTransformation)
                for trans in transformation.transformations
            )
        ):
            return self.properties_log_x
        elif isinstance(transformation, IdentityTransformation) or (
            isinstance(transformation, ComposedTransformation)
            and all(
                isinstance(trans, IdentityTransformation)
                for trans in transformation.transformations
            )
        ):
            return self.properties

        else:
            raise TypeError(
                "The transformation provided is not compatible with pybop.MultivariateLogNormal. "
                "Only pybop.LogTransformation and pybop.IdentityTransformation are allowed."
            )


class MarginalDistribution(Distribution):
    """
    Represents a univariate marginal distribution of a pybop.BaseMultivariateDistribution.
    Relies on the "marginal" method of the multivariate distribution.

    Sub-class of pybop.Distribution with the additional properties
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
        super().__init__(distribution)

        # add additional properties
        self._position = position
        self.parent_distribution = parent_distribution

    def mean(self):
        if isinstance(self.parent_distribution, MultivariateNonparametric):
            # scipy.stats.gaussian_kde does not have the method "mean"
            # compute mean based on dataset
            return np.mean(self.distribution.dataset)
        else:
            return self.distribution.mean()

    def std(self):
        if isinstance(self.parent_distribution, MultivariateNonparametric):
            # scipy.stats.gaussian_kde does not have the method "std"
            # compute standard deviation based on dataset
            return np.std(self.distribution.dataset, ddof=1)
        else:
            return self.distribution.std()

    @property
    def position(self):
        return self._position
