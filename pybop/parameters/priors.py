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
    distribution : scipy.stats.rv_continuous
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
        return self.distribution.pdf(x, loc=self.loc, scale=self.scale)

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
        return self.distribution.logpdf(x, loc=self.loc, scale=self.scale)

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
        return self.distribution.ppf(q, loc=self.loc, scale=self.scale)

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
        return self.distribution.cdf(x, loc=self.loc, scale=self.scale)

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
        if not isinstance(size, int | tuple):
            raise ValueError(
                "size must be a positive integer or tuple of positive integers"
            )
        if isinstance(size, int) and size < 1:
            raise ValueError("size must be a positive integer")
        if isinstance(size, tuple) and any(s < 1 for s in size):
            raise ValueError("size must be a tuple of positive integers")

        return self.distribution.rvs(
            loc=self.loc, scale=self.scale, size=size, random_state=random_state
        )

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
        x = self.verify(x)
        return self.logpdf(x), self._dlogpdf_dx(x)

    def _dlogpdf_dx(self, x):
        """
        Evaluates the first derivative of the log distribution at x.

        Overwrite this function in a subclass to improve upon this generic
        finite difference approximation.

        Parameters
        ----------
        x : float
            The point(s) at which to evaluate the first derivative.

        Returns
        -------
        float
            The value(s) of the first derivative at x.
        """
        # Use a finite difference approximation of the gradient
        delta = max(abs(x) * 1e-3, np.finfo(float).eps)
        log_prior_upper = self.joint_prior.logpdf(x + delta)
        log_prior_lower = self.joint_prior.logpdf(x - delta)

        return (log_prior_upper - log_prior_lower) / (2 * delta)

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
        """Return a string representation of the object."""
        return f"{self.__class__.__name__}, loc: {self.loc}, scale: {self.scale}"

    @property
    def mean(self):
        """The mean of the distribution."""
        return self.distribution.mean(loc=self.loc, scale=self.scale)

    @property
    def sigma(self):
        """The standard deviation of the distribution."""
        return self.distribution.std(loc=self.loc, scale=self.scale)

    def bounds(self) -> tuple[float, float] | None:
        """Get the bounds of the distribution, if any."""
        upper = self.distribution.ppf(1, loc=self.loc, scale=self.scale)
        lower = self.distribution.ppf(0, loc=self.loc, scale=self.scale)
        if np.isinf(upper) and np.isinf(lower):
            return None
        return (lower, upper)


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
        self.distribution = stats.norm
        self._n_parameters = 1

    def _dlogpdf_dx(self, x):
        """
        Evaluates the first derivative of the log Gaussian distribution at x.

        Parameters
        ----------
        x : float
            The point(s) at which to evaluate the first derivative.

        Returns
        -------
        float
            The value(s) of the first derivative at x.
        """
        return (self.loc - x) / self.scale**2


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
        self.distribution = stats.uniform
        self._n_parameters = 1

    def _dlogpdf_dx(self, x):
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
        return np.zeros_like(x)


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

    def __init__(self, loc=0, scale=1, random_state=None):
        super().__init__()
        self.name = "Exponential"
        self.loc = loc
        self.scale = scale
        self.distribution = stats.expon
        self._n_parameters = 1

    def _dlogpdf_dx(self, x):
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
        return -1 / self.scale * np.ones_like(x)


class JointPrior(BasePrior):
    """
    Represents a joint prior distribution composed of multiple prior distributions.

    Parameters
    ----------
    priors : BasePrior
        One or more prior distributions to combine into a joint distribution.
    """

    def __init__(self, *priors: BasePrior):
        super().__init__()

        if all(prior is None for prior in priors):
            return

        if not all(isinstance(prior, BasePrior) for prior in priors):
            raise ValueError("All priors must be instances of BasePrior")

        self._n_parameters = len(priors)
        self._priors: list[BasePrior] = list(priors)

    def logpdf(self, x: float | np.ndarray) -> float:
        """
        Evaluates the joint log-prior distribution at a given point.

        Parameters
        ----------
        x : float | np.ndarray
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

        return sum(prior.logpdf(x) for prior, x in zip(self._priors, x, strict=False))

    def logpdfS1(self, x: float | np.ndarray) -> tuple[float, np.ndarray]:
        """
        Evaluates the first derivative of the joint log-prior distribution at a given point.

        Parameters
        ----------
        x : float | np.ndarray
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

        for prior, xi in zip(self._priors, x, strict=False):
            p, dp = prior.logpdfS1(xi)
            log_probs.append(p)
            derivatives.append(dp)

        output = sum(log_probs)
        doutput = np.asarray(derivatives)

        if doutput.ndim == 1:
            return output, doutput

        return output, doutput.T

    def __repr__(self) -> str:
        priors_repr = ", ".join([repr(prior) for prior in self._priors])
        return f"{self.__class__.__name__}(priors: [{priors_repr}])"
