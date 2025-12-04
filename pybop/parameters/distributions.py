import numpy as np
import scipy.stats as stats


class Distribution:
    """
    A base class for defining parameter distributions.

    This class provides a foundation for implementing various distributions.
    It includes methods for calculating the probability density function (PDF),
    log probability density function (log PDF), and generating random variates
    from the distribution.

    Attributes
    ----------
    distribution : scipy.stats.distributions.rv_frozen
        The underlying continuous random variable distribution.
    """

    def __init__(
        self,
        distribution: stats._distribution_infrastructure.ContinuousDistribution
        | None = None,
    ):
        self.distribution = distribution

    def support(self):
        return self.distribution.support()

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
        if self.distribution is None:
            raise NotImplementedError
        else:
            return self.distribution.pdf(x)

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
        if self.distribution is None:
            raise NotImplementedError
        else:
            return self.distribution.logpdf(x)

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
        if self.distribution is None:
            raise NotImplementedError
        else:
            return self.distribution.icdf(q)

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
        if self.distribution is None:
            raise NotImplementedError
        else:
            return self.distribution.cdf(x)

    def sample(self, shape=1, rng=None):
        """
        Generates random variates from the distribution.

        Parameters
        ----------
        shape : int
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
            If the shape parameter is negative.
        """
        if not isinstance(shape, int | tuple):
            raise ValueError(
                "shape must be a positive integer or tuple of positive integers"
            )
        if isinstance(shape, int) and shape < 1:
            raise ValueError("shape must be a positive integer")
        if isinstance(shape, tuple) and any(s < 1 for s in shape):
            raise ValueError("shape must be a tuple of positive integers")

        if self.distribution is None:
            raise NotImplementedError
        else:
            return self.distribution.sample(shape=shape, rng=rng)

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
        if self.distribution is None:
            raise NotImplementedError
        else:
            # Use a finite difference approximation of the gradient
            delta = max(abs(x) * 1e-3, np.finfo(float).eps)
            log_distribution_upper = self.logpdf(x + delta)
            log_distribution_lower = self.logpdf(x - delta)

            return (log_distribution_upper - log_distribution_lower) / (2 * delta)

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
        return f"{self.__class__.__name__}, mean: {self.mean()}, standard deviation: {self.standard_deviation()}"

    def mean(self):
        """
        Get the mean of the distribution.

        Returns
        -------
        float
            The mean of the distribution.
        """
        return self.distribution.mean()

    def standard_deviation(self):
        """
        Get the standard deviation of the distribution.

        Returns
        -------
        float
            The standard deviation of the distribution.
        """
        return self.distribution.standard_deviation()


class Gaussian(Distribution):
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

    def __init__(
        self,
        mean,
        sigma,
        truncated_at: list[float] = None,
    ):
        distribution = stats.Normal(mu=mean, sigma=sigma)
        if truncated_at is not None:
            distribution = stats.truncate(
                distribution, truncated_at[0], truncated_at[1]
            )
        super().__init__(distribution)
        self.name = "Gaussian"
        self._n_parameters = 1
        self.loc = mean
        self.scale = sigma

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


class Uniform(Distribution):
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

    def __init__(
        self,
        lower,
        upper,
    ):
        super().__init__(stats.Uniform(a=lower, b=upper))
        self.name = "Uniform"
        self.lower = lower
        self.upper = upper
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

    def mean(self):
        """
        Returns the mean of the distribution.
        """
        return (self.upper - self.lower) / 2

    def __repr__(self):
        """
        Returns a string representation of the object.
        """
        return f"{self.__class__.__name__}, lower: {self.lower}, upper: {self.upper}"


class Exponential(Distribution):
    """
    Represents an exponential distribution with a specified scale parameter.

    This class provides methods to calculate the pdf, the log pdf, and to generate random
    variates from the distribution.

    Parameters
    ----------
    scale : float
        The scale parameter (lambda) of the exponential distribution.
    """

    def __init__(
        self,
        scale: float,
        loc: float = 0,
    ):
        distr = stats.make_distribution(stats.expon)
        X = distr()
        super().__init__(scale * X + loc)
        self.name = "Exponential"
        self._n_parameters = 1
        self.loc = loc
        self.scale = scale

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

    def __repr__(self):
        """
        Returns a string representation of the object.
        """
        return f"{self.__class__.__name__}, loc: {self.loc}, scale: {self.scale}"


class JointDistribution(Distribution):
    """
    Represents a joint distribution distribution composed of multiple distribution distributions.

    Parameters
    ----------
    distributions : Distribution
        One or more distribution distributions to combine into a joint distribution.
    """

    def __init__(self, *distributions: Distribution | stats.distributions.rv_frozen):
        super().__init__()

        if all(distribution is None for distribution in distributions):
            return

        if not all(
            isinstance(
                distribution,
                (
                    Distribution,
                    stats._distribution_infrastructure.ContinuousDistribution,  # noqa SLF001
                ),
            )
            for distribution in distributions
        ):
            raise ValueError(
                "All distributions must be instances of Distribution or scipy.stats.distributions.rv_frozen"
            )

        self._n_parameters = len(distributions)
        self._distributions: list[Distribution] = [
            distribution
            if isinstance(distribution, Distribution)
            else Distribution(distribution)
            for distribution in distributions
        ]

    def logpdf(self, x: float | np.ndarray) -> float:
        """
        Evaluates the joint log-distribution distribution at a given point.

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

        return sum(
            distribution.logpdf(x)
            for distribution, x in zip(self._distributions, x, strict=False)
        )

    def logpdfS1(self, x: float | np.ndarray) -> tuple[float, np.ndarray]:
        """
        Evaluates the first derivative of the joint log-distribution distribution at a given point.

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

        for distribution, xi in zip(self._distributions, x, strict=False):
            p, dp = distribution.logpdfS1(xi)
            log_probs.append(p)
            derivatives.append(dp)

        output = sum(log_probs)
        doutput = np.asarray(derivatives)

        if doutput.ndim == 1:
            return output, doutput

        return output, doutput.T

    def __repr__(self) -> str:
        distributions_repr = "; ".join(
            [repr(distribution) for distribution in self._distributions]
        )
        return f"{self.__class__.__name__}(distributions: [{distributions_repr}])"
