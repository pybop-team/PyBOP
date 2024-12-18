import numpy as np
import scipy.integrate as integrate
import scipy.stats as stats
from scipy.linalg import sqrtm

from pybop import BasePrior


def insert_x_at_dim(other, x, dim):
    """
    Inserts each element of x at dim in other.

    Parameters
    ----------
    other : numpy.ndarray
        The fixed other variables in all dimensions but dim.
    x : numpy.ndarray
        The point(s) at which to evaluate in dim.
    dim : int
        The dimension at which to insert x.
    """
    x = np.atleast_1d(x)
    return np.array(
        [np.concatenate(other[: dim - 1], [entry], other[dim - 1 :]) for entry in x]
    )


class BaseMultivariatePrior(BasePrior):
    """
    A base class for defining multivariate prior distributions.

    This class extends ``pybop.BasePrior`` with methods that reduce the
    output of a multivariate distribution to its individual dimensions.

    Note that, unlinke in ``pybop.BasePrior``, distribution attributes
    are stored in a dictionary, as multivariate distributiosn in SciPy
    do not follow the loc/scale convention.

    Attributes
    ----------
    prior : scipy.stats.rv_continuous
        The underlying continuous random variable distribution.
    properties : dict
        A dictionary with distribution keyword argument names as string
        keys and their values as float values.
    """

    def pdf(self, x):
        return self.prior.pdf(x, **self.properties)

    def pdf_marginal(self, x, dim):
        """
        Integrates the probability density function (PDF) at x down to
        one variable.

        Parameters
        ----------
        x : numpy.ndarray
            The point(s) at which to evaluate the pdf.
        dim : int
            The dimension to which to reduce the pdf to.

        Returns
        -------
        float
            The marginal probability density function value at x in dim.
        """
        unnormalized_pdf = integrate.nquad(
            lambda y: self.pdf(insert_x_at_dim(y, x, dim)),
            [
                [-np.inf, np.inf] * (dim - 1)
                + [-np.inf, np.inf] * (self.prior.dim - dim)
            ],
        )[0]
        return unnormalized_pdf / np.sum(unnormalized_pdf)

    def logpdf(self, x):
        return self.prior.logpdf(x, **self.properties)

    def logpdf_marginal(self, x, dim):
        """
        Integrates the logarithm of the probability density function
        (PDF) at x down to one variable.

        Parameters
        ----------
        x : numpy.ndarray
            The point(s) at which to evaluate the PDF.
        dim : int
            The dimension to which to reduce the PDF to.

        Returns
        -------
        float
            The log marginal probability density function value at x in
            dim.
        """
        return np.log(self.pdf_marginal(x, dim))

    icdf = None
    """Multivariate distributions have no invertible CDF."""

    def icdf_marginal(self, q, dim):
        raise NotImplementedError

    def cdf(self, x):
        return self.prior.cdf(x, **self.properties)

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
        return self.prior.rvs(**self.properties, size=size, random_state=random_state)

    def __repr__(self):
        return f"{self.__class__.__name__}, properties: {self.properties}"

    @property
    def mean(self):
        raise NotImplementedError

    @property
    def sigma(self):
        raise NotImplementedError


class MultivariateGaussian(BaseMultivariatePrior):
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
    """

    def __init__(self, mean, covariance, random_state=None):
        super().__init__()
        self.name = "MultivariateGaussian"
        self.prior = stats.multivariate_normal
        self.properties = {"mean": mean, "cov": covariance}
        self.mean = mean
        self.sigma = sqrtm(covariance)
        self.sigma2 = covariance
        self._n_parameters = len(mean)
