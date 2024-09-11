from typing import Union

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

    def icdf(self, q):
        """
        Calculates the inverse cumulative distribution function (CDF) of the distribution at q.

        Parameters
        ----------
        q : float
            The point(s) at which to evaluate the inverse CDF.

        Returns
        -------
        float
            The inverse cumulative distribution function value at q.
        """
        return self.prior.ppf(q, scale=self.scale, loc=self.loc)

    def cdf(self, x):
        """
        Calculates the cumulative distribution function (CDF) of the distribution at x.

        Parameters
        ----------
        x : float
            The point(s) at which to evaluate the CDF.

        Returns
        -------
        float
            The cumulative distribution function value at x.
        """
        return self.prior.cdf(x, scale=self.scale, loc=self.loc)

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

    def __call__(self, x):
        """
        Evaluates the distribution at x.

        Parameters
        ----------
        x : float
            The point(s) at which to evaluate the distribution.

        Returns
        -------
        float
            The value(s) of the distribution at x.
        """
        inputs = self.verify(x)
        return self.logpdf(inputs)

    def logpdfS1(self, x):
        """
        Evaluates the first derivative of the distribution at x.

        Parameters
        ----------
        x : float
            The point(s) at which to evaluate the first derivative.

        Returns
        -------
        float
            The value(s) of the first derivative at x.
        """
        inputs = self.verify(x)
        return self._logpdfS1(inputs)

    def _logpdfS1(self, x):
        """
        Evaluates the first derivative of the distribution at x.

        Parameters
        ----------
        x : float
            The point(s) at which to evaluate the first derivative.

        Returns
        -------
        float
            The value(s) of the first derivative at x.
        """
        raise NotImplementedError

    def verify(self, x):
        """
        Verifies that the input is a numpy array and converts it if necessary.
        """
        if isinstance(x, dict):
            x = np.asarray(list(x.values()))
        elif not isinstance(x, np.ndarray):
            x = np.asarray(x)
        return x

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
        super().__init__()
        self.name = "Gaussian"
        self.loc = mean
        self.scale = sigma
        self.prior = stats.norm
        self._offset = -0.5 * np.log(2 * np.pi * self.scale**2)
        self.sigma2 = self.scale**2
        self._multip = -1 / (2.0 * self.sigma2)
        self._n_parameters = 1

    def _logpdfS1(self, x):
        """
        Evaluates the first derivative of the gaussian (log) distribution at x.

        Parameters
        ----------
        x : float
            The point(s) at which to evaluate the first derivative.

        Returns
        -------
        float
            The value(s) of the first derivative at x.
        """
        return self.__call__(x), -(x - self.loc) * self._multip


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
        super().__init__()
        self.name = "Uniform"
        self.lower = lower
        self.upper = upper
        self.loc = lower
        self.scale = upper - lower
        self.prior = stats.uniform
        self._n_parameters = 1

    def _logpdfS1(self, x):
        """
        Evaluates the first derivative of the log uniform distribution at x.

        Parameters
        ----------
        x : float
            The point(s) at which to evaluate the first derivative.

        Returns
        -------
        float
            The value(s) of the first derivative at x.
        """
        log_pdf = self.__call__(x)
        dlog_pdf = np.zeros_like(x)
        return log_pdf, dlog_pdf

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
        super().__init__()
        self.name = "Exponential"
        self.loc = loc
        self.scale = scale
        self.prior = stats.expon
        self._n_parameters = 1

    def _logpdfS1(self, x):
        """
        Evaluates the first derivative of the log exponential distribution at x.

        Parameters
        ----------
        x : float
            The point(s) at which to evaluate the first derivative.

        Returns
        -------
        float
            The value(s) of the first derivative at x.
        """
        log_pdf = self.__call__(x)
        dlog_pdf = -1 / self.scale * np.ones_like(x)
        return log_pdf, dlog_pdf


class JointLogPrior(BasePrior):
    """
    Represents a joint prior distribution composed of multiple prior distributions.

    Parameters
    ----------
    priors : BasePrior
        One or more prior distributions to combine into a joint distribution.
    """

    def __init__(self, *priors: BasePrior):
        super().__init__()

        if not all(isinstance(prior, BasePrior) for prior in priors):
            raise ValueError("All priors must be instances of BasePrior")

        self._n_parameters = len(priors)
        self._priors: list[BasePrior] = list(priors)

    def logpdf(self, x: Union[float, np.ndarray]) -> float:
        """
        Evaluates the joint log-prior distribution at a given point.

        Parameters
        ----------
        x : Union[float, np.ndarray]
            The point(s) at which to evaluate the distribution. The length of `x`
            should match the total number of parameters in the joint distribution.

        Returns
        -------
        float
            The joint log-probability density of the distribution at `x`.
        """
        if len(x) != self._n_parameters:
            raise ValueError(
                f"Input x must have length {self._n_parameters}, got {len(x)}"
            )

        return sum(prior(x) for prior, x in zip(self._priors, x))

    def _logpdfS1(self, x: Union[float, np.ndarray]) -> tuple[float, np.ndarray]:
        """
        Evaluates the first derivative of the joint log-prior distribution at a given point.

        Parameters
        ----------
        x : Union[float, np.ndarray]
            The point(s) at which to evaluate the first derivative. The length of `x`
            should match the total number of parameters in the joint distribution.

        Returns
        -------
        Tuple[float, np.ndarray]
            A tuple containing the log-probability density and its first derivative at `x`.
        """
        if len(x) != self._n_parameters:
            raise ValueError(
                f"Input x must have length {self._n_parameters}, got {len(x)}"
            )

        log_probs = []
        derivatives = []

        for prior, xi in zip(self._priors, x):
            p, dp = prior.logpdfS1(xi)
            log_probs.append(p)
            derivatives.append(dp)

        output = sum(log_probs)
        doutput = np.array(derivatives)

        return output, doutput

    def __repr__(self) -> str:
        priors_repr = ", ".join([repr(prior) for prior in self._priors])
        return f"{self.__class__.__name__}(priors: [{priors_repr}])"
