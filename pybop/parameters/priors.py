import numpy as np
import scipy.stats as stats

from pybop import Bounds, Parameter
from pybop.transformation.base_transformation import Transformation


class ParameterDistribution(Parameter):
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
        distribution: stats.rv_continuous = None,
        initial_value: float = None,
        transformation: Transformation | None = None,
        margin: float = 1e-4,
    ):
        super().__init__(
            initial_value=initial_value, transformation=transformation, margin=margin
        )
        self._distribution = distribution
        if initial_value is None and distribution is not None:
            initial_value = self.sample_from_distribution()[0]
        self._initial_value = (
            float(initial_value) if initial_value is not None else None
        )
        self._current_value = self._initial_value

        if distribution is not None:
            lower, upper = self._distribution.support()
            self._bounds = Bounds(lower, upper)

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
        if self._distribution is None:
            raise NotImplementedError
        else:
            return self._distribution.pdf(x)

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
        if self._distribution is None:
            raise NotImplementedError
        else:
            return self._distribution.logpdf(x)

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
        if self._distribution is None:
            raise NotImplementedError
        else:
            return self._distribution.ppf(q)

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
        if self._distribution is None:
            raise NotImplementedError
        else:
            return self._distribution.cdf(x)

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

        if self._distribution is None:
            raise NotImplementedError
        else:
            return self._distribution.rvs(size=size, random_state=random_state)

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
        if self._distribution is None:
            raise NotImplementedError
        else:
            # Use a finite difference approximation of the gradient
            delta = max(abs(x) * 1e-3, np.finfo(float).eps)
            log_prior_upper = self.logpdf(x + delta)
            log_prior_lower = self.logpdf(x - delta)

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
        """
        Returns a string representation of the object.
        """
        return f"{self.__class__.__name__}, mean: {self.mean}, standard deviation: {self.sigma}"

    @property
    def mean(self):
        """
        Get the mean of the distribution.

        Returns
        -------
        float
            The mean of the distribution.
        """
        return self._distribution.mean()

    @property
    def sigma(self):
        """
        Get the standard deviation of the distribution.

        Returns
        -------
        float
            The standard deviation of the distribution.
        """
        return self._distribution.std()


class Gaussian(ParameterDistribution):
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
        bounds: list[float] = None,
        initial_value: float = None,
        transformation: Transformation | None = None,
        margin: float = 1e-4,
    ):
        if bounds is not None:
            distribution = stats.truncnorm(
                (bounds[0] - mean) / sigma,
                (bounds[1] - mean) / sigma,
                loc=mean,
                scale=sigma,
            )
        else:
            distribution = stats.norm(loc=mean, scale=sigma)
        ParameterDistribution.__init__(
            self,
            distribution,
            initial_value=initial_value,
            transformation=transformation,
            margin=margin,
        )
        self.name = "Gaussian"
        self._n_parameters = 1
        self.loc = mean
        self.scale = sigma
        if bounds is not None:
            self._bounds = Bounds(bounds[0], bounds[1])
        else:
            self._bounds = None

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


class Uniform(ParameterDistribution):
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
        initial_value: float = None,
        transformation: Transformation | None = None,
        margin: float = 1e-4,
    ):
        ParameterDistribution.__init__(
            self,
            stats.uniform(loc=lower, scale=upper - lower),
            initial_value=initial_value,
            transformation=transformation,
            margin=margin,
        )
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

    def __repr__(self):
        """
        Returns a string representation of the object.
        """
        return f"{self.__class__.__name__}, lower: {self.lower}, upper: {self.upper}"


class Exponential(ParameterDistribution):
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
        initial_value: float = None,
        transformation: Transformation | None = None,
        margin: float = 1e-4,
    ):
        ParameterDistribution.__init__(
            self,
            stats.expon(loc=loc, scale=scale),
            initial_value=initial_value,
            transformation=transformation,
            margin=margin,
        )
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


class JointPrior(ParameterDistribution):
    """
    Represents a joint prior distribution composed of multiple prior distributions.

    Parameters
    ----------
    priors : ParameterDistribution
        One or more prior distributions to combine into a joint distribution.
    """

    def __init__(self, *priors: ParameterDistribution | stats.distributions.rv_frozen):
        super().__init__()

        if all(prior is None for prior in priors):
            return

        if not all(
            isinstance(prior, (ParameterDistribution, stats.distributions.rv_frozen))
            for prior in priors
        ):
            raise ValueError("All priors must be instances of ParameterDistribution")

        self._n_parameters = len(priors)
        self._priors: list[ParameterDistribution] = [
            prior
            if isinstance(prior, ParameterDistribution)
            else ParameterDistribution(prior)
            for prior in priors
        ]

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
