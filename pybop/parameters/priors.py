import numpy as np
import scipy.stats as stats


class BasePrior:
    """
    A base class for defining prior distributions.

    This class provides a foundation for implementing various prior distributions.
    It includes methods for calculating the probability density function (PDF),
    log probability density function (log PDF), and generating random variates
    from the distribution.

    Attributes
    ----------
    prior : scipy.stats.rv_continuous
        The underlying continuous random variable distribution.
    loc : float
        The location parameter of the distribution.
    scale : float
        The scale parameter of the distribution.
    """

    def __init__(self):
        pass

    def pdf(self, x):
        """
        Calculates the probability density function (PDF) of the distribution at x.

        Parameters
        ----------
        x : float
            The point(s) at which to evaluate the pdf.

        Returns
        -------
        float
            The probability density function value at x.
        """
        return self.prior.pdf(x, loc=self.loc, scale=self.scale)

    def logpdf(self, x):
        """
        Calculates the logarithm of the probability density function of the distribution at x.

        Parameters
        ----------
        x : float
            The point(s) at which to evaluate the log pdf.

        Returns
        -------
        float
            The logarithm of the probability density function value at x.
        """
        return self.prior.logpdf(x, loc=self.loc, scale=self.scale)

    def rvs(self, size=1, random_state=None):
        """
        Generates random variates from the distribution.

        Parameters
        ----------
        size : int
            The number of random variates to generate.
        random_state : int, optional
            The random state seed for reproducibility. Default is None.

        Returns
        -------
        array_like
            An array of random variates from the distribution.

        Raises
        ------
        ValueError
            If the size parameter is negative.
        """
        if not isinstance(size, (int, tuple)):
            raise ValueError(
                "size must be a positive integer or tuple of positive integers"
            )
        if isinstance(size, int) and size < 1:
            raise ValueError("size must be a positive integer")
        if isinstance(size, tuple) and any(s < 1 for s in size):
            raise ValueError("size must be a tuple of positive integers")

        return self.prior.rvs(
            loc=self.loc, scale=self.scale, size=size, random_state=random_state
        )

    def __repr__(self):
        """
        Returns a string representation of the object.
        """
        return f"{self.__class__.__name__}, loc: {self.loc}, scale: {self.scale}"

    @property
    def mean(self):
        """
        Get the mean of the distribution.

        Returns
        -------
        float
            The mean of the distribution.
        """
        return self.loc

    @property
    def sigma(self):
        """
        Get the standard deviation of the distribution.

        Returns
        -------
        float
            The standard deviation of the distribution.
        """
        return self.scale


class Gaussian(BasePrior):
    """
    Represents a Gaussian (normal) distribution with a given mean and standard deviation.

    This class provides methods to calculate the probability density function (pdf),
    the logarithm of the pdf, and to generate random variates (rvs) from the distribution.

    Parameters
    ----------
    mean : float
        The mean (mu) of the Gaussian distribution.
    sigma : float
        The standard deviation (sigma) of the Gaussian distribution.
    """

    def __init__(self, mean, sigma, random_state=None):
        self.name = "Gaussian"
        self.loc = mean
        self.scale = sigma
        self.prior = stats.norm


class Uniform(BasePrior):
    """
    Represents a uniform distribution over a specified interval.

    This class provides methods to calculate the pdf, the log pdf, and to generate
    random variates from the distribution.

    Parameters
    ----------
    lower : float
        The lower bound of the distribution.
    upper : float
        The upper bound of the distribution.
    """

    def __init__(self, lower, upper, random_state=None):
        self.name = "Uniform"
        self.lower = lower
        self.upper = upper
        self.loc = lower
        self.scale = upper - lower
        self.prior = stats.uniform

    @property
    def mean(self):
        """
        Returns the mean of the distribution.
        """
        return (self.upper - self.lower) / 2

    @property
    def sigma(self):
        """
        Returns the standard deviation of the distribution.
        """
        return (self.upper - self.lower) / (2 * np.sqrt(3))


class Exponential(BasePrior):
    """
    Represents an exponential distribution with a specified scale parameter.

    This class provides methods to calculate the pdf, the log pdf, and to generate random
    variates from the distribution.

    Parameters
    ----------
    scale : float
        The scale parameter (lambda) of the exponential distribution.
    """

    def __init__(self, scale, loc=0, random_state=None):
        self.name = "Exponential"
        self.loc = loc
        self.scale = scale
        self.prior = stats.expon
