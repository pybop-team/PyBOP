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
        return self.evaluate(x)

    def evaluate(self, x):
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
        return self._evaluate(inputs)

    def _evaluate(self, x):
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
        return self.logpdf(x)

    def evaluateS1(self, x):
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
        return self._evaluateS1(inputs)

    def _evaluateS1(self, x):
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

    @property
    def n_parameters(self):
        return self._n_parameters


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
        self._offset = -0.5 * np.log(2 * np.pi * self.scale**2)
        self.sigma2 = self.scale**2
        self._multip = -1 / (2.0 * self.sigma2)
        self._n_parameters = 1

    def _evaluateS1(self, x):
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
        if not isinstance(x, np.ndarray):
            x = np.asarray(x)
        return self(x), -(x - self.loc) * self._multip


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
        self._n_parameters = 1

    def _evaluateS1(self, x):
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
        self.name = "Exponential"
        self.loc = loc
        self.scale = scale
        self.prior = stats.expon
        self._n_parameters = 1

    def _evaluateS1(self, x):
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


class ComposedLogPrior(BasePrior):
    """
    Represents a composition of multiple prior distributions.
    """

    def __init__(self, *priors):
        self._priors = priors
        for prior in priors:
            if not isinstance(prior, BasePrior):
                raise ValueError("All priors must be instances of BasePrior")

        self._n_parameters = len(priors)  # Needs to be updated

    def _evaluate(self, x):
        """
        Evaluates the composed prior distribution at x.

        Parameters
        ----------
        x : float
            The point(s) at which to evaluate the distribution.

        Returns
        -------
        float
            The value(s) of the distribution at x.
        """
        return sum(prior(x) for prior, x in zip(self._priors, x))

    def _evaluateS1(self, x):
        """
        Evaluates the first derivative of the composed prior distribution at x.
        Inspired by PINTS implementation.

        *This method only works if the underlying :class:`LogPrior` classes all
        implement the optional method :class:`LogPDF.evaluateS1().`.*

        Parameters
        ----------
        x : float
            The point(s) at which to evaluate the first derivative.

        Returns
        -------
        float
            The value(s) of the first derivative at x.
        """
        output = 0
        doutput = np.zeros(self.n_parameters)
        index = 0

        for prior in self._priors:
            num_params = prior.n_parameters
            x_subset = x[index : index + num_params]
            p, dp = prior.evaluateS1(x_subset)
            output += p
            doutput[index : index + num_params] = dp
            index += num_params

        return output, doutput

    def __repr__(self):
        return f"{self.__class__.__name__}, priors: {self._priors}"
