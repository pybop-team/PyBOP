import numpy as np
import scipy.stats as stats

from pybop.transformation.base_transformation import Transformation
from pybop.transformation.transformations import (
    IdentityTransformation,
    LogTransformation,
    ScaledTransformation,
)


class BaseDistribution:
    """
    A base class for defining parameter distributions.

    This class provides a foundation for implementing various distributions.
    It includes methods for calculating the probability density function (PDF),
    log probability density function (log PDF), and generating random variates
    from the distribution.

    Attributes
    ----------
    properties : dict
        A dictionary with distribution keyword argument names as string
        keys and their values as float values.
    n_parameters : int
        The number of dimensions (default: 1).
    """

    def __init__(self, properties: dict | None = None, n_parameters: int = 1):
        self.properties = properties or {}
        self._n_parameters = n_parameters

    def support(self) -> tuple[float]:
        """Returns the support of the distribution, to be overwritten by child classes."""
        return (-np.inf, np.inf)

    def mean(self) -> float:
        """Get the mean of the distribution."""
        raise NotImplementedError

    def std(self) -> float:
        """Get the standard deviation of the distribution."""
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

    def rvs(self, size: int = 1, random_state: int | None = None):
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
        raise NotImplementedError

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
        log_distribution_upper = self.logpdf(x + delta)
        log_distribution_lower = self.logpdf(x - delta)

        return (log_distribution_upper - log_distribution_lower) / (2 * delta)

    def verify(self, x) -> np.ndarray:
        """Verifies that the input is a numpy array and converts it if necessary."""
        if isinstance(x, dict):
            x = np.asarray(list(x.values()))
        elif not isinstance(x, np.ndarray):
            x = np.asarray(x)
        return x

    def __repr__(self) -> str:
        return f"{self.name}, properties: {self.properties}"

    def get_transformed_distribution(self, transform: Transformation):
        """Get the transformed distribution in the search space."""
        if isinstance(transform, IdentityTransformation):
            return self

        transformed_distribution = self._transform(transform)

        if transformed_distribution is None:
            raise TypeError(
                f"The transformation of a {self.name} distribution by a "
                f"{transform.name} is undefined or not yet implemented."
            )
        return transformed_distribution

    def _transform(self, transform: Transformation):
        """Get the transformed distribution in the search space."""
        return NotImplementedError

    @property
    def name(self):
        return self.__class__.__name__


class Distribution(BaseDistribution):
    """
    A base class for distributions based on a scipy.stats distribution.

    This class provides a foundation for implementing various distributions.
    It includes methods for calculating the probability density function (PDF),
    log probability density function (log PDF), and generating random variates
    from the distribution.

    Additional Attributes
    ---------------------
    distribution : scipy.stats.distributions.rv_frozen
        The underlying continuous random variable distribution.
    """

    def __init__(
        self,
        distribution: stats.distributions.rv_frozen,
        properties: dict | None = None,
        n_parameters: int = 1,
    ):
        super().__init__(properties=properties, n_parameters=n_parameters)
        self.distribution = distribution

    def support(self) -> tuple[float]:
        return tuple(float(x) for x in self.distribution.support())

    def pdf(self, x):
        return self.distribution.pdf(x)

    def logpdf(self, x):
        return self.distribution.logpdf(x)

    def icdf(self, q):
        return self.distribution.ppf(q)

    def cdf(self, x):
        return self.distribution.cdf(x)

    def rvs(self, size: int = 1, random_state: int | None = None):
        return self.distribution.rvs(size=size, random_state=random_state)

    def mean(self) -> float:
        return self.distribution.mean()

    def std(self) -> float:
        return self.distribution.std()


class Gaussian(Distribution):
    """
    Represents a Gaussian (normal) distribution with a given mean and standard deviation.

    Parameters
    ----------
    mean : float
        The mean (mu) of the Gaussian distribution.
    sigma : float
        The standard deviation (sigma) of the Gaussian distribution.
    truncated_at : list[float], optional
        If provided, the distribution becomes a truncated normal distribution.
    """

    def __init__(
        self, mean: float, sigma: float, truncated_at: list[float] | None = None
    ):
        if truncated_at is not None:
            properties = {
                "a": float((truncated_at[0] - mean) / sigma),
                "b": float((truncated_at[1] - mean) / sigma),
                "loc": float(mean),
                "scale": float(sigma),
            }
            distribution = stats.truncnorm(**properties)
        else:
            properties = {"loc": float(mean), "scale": float(sigma)}
            distribution = stats.norm(**properties)
        super().__init__(distribution, properties=properties)

    def _dlogpdf_dx(self, x):
        """Evaluates the first derivative of the log Gaussian distribution at x."""
        if "a" in self.properties.keys():
            return super()._dlogpdf_dx(x)  # use estimate for a truncated normal

        return (self.properties["loc"] - x) / self.properties["scale"] ** 2

    def _transform(self, transform: Transformation) -> BaseDistribution | None:
        """Get the transformed distribution in the search space."""
        if isinstance(transform, ScaledTransformation):
            truncated_at = None
            if "a" in self.properties.keys():
                original_truncation = self.properties["loc"] + (
                    self.properties["scale"]
                    * np.asarray([self.properties["a"], self.properties["b"]])
                )
                truncated_at = np.sort(
                    [transform.to_search(x).item() for x in original_truncation]
                )
            return Gaussian(
                mean=transform.to_search(self.properties["loc"]).item(),
                sigma=self.properties["scale"] * transform.coefficient.item(),
                truncated_at=truncated_at,
            )
        return None


class LogNormal(Distribution):
    """
    Represents a log-normal distribution corresponding to a Gaussian distribution for
    log(x) with a given mean and standard deviation.

    Parameters
    ----------
    mean_log_x : float
        The mean (mu) of the Gaussian distribution of log(x).
    sigma : float
        The standard deviation (sigma) of the Gaussian distribution of log(x).
    """

    def __init__(self, mean_log_x: float, sigma: float):
        properties = {"scale": float(np.exp(mean_log_x)), "s": float(sigma)}
        super().__init__(stats.lognorm(**properties), properties=properties)

    def _transform(self, transform: Transformation) -> BaseDistribution | None:
        """Get the transformed distribution in the search space."""
        if isinstance(transform, LogTransformation):
            return Gaussian(
                mean=np.log(self.properties["scale"]),
                sigma=self.properties["s"],
            )
        return None


class Uniform(Distribution):
    """
    Represents a uniform distribution over a specified interval.

    Parameters
    ----------
    lower : float
        The lower bound of the distribution.
    upper : float
        The upper bound of the distribution.
    """

    def __init__(self, lower: float, upper: float):
        properties = {"loc": float(lower), "scale": float(upper - lower)}
        super().__init__(stats.uniform(**properties), properties=properties)

    def _dlogpdf_dx(self, x):
        """Evaluates the first derivative of the log uniform distribution at x."""
        return np.zeros_like(x)

    def mean(self) -> float:
        """Returns the mean of the distribution."""
        return self.properties["loc"] + self.properties["scale"] / 2

    def __repr__(self) -> str:
        return f"{self.name}, bounds: {self.support()}"

    def _transform(self, transform: Transformation) -> BaseDistribution | None:
        """Get the transformed distribution in the search space."""
        if isinstance(transform, ScaledTransformation):
            bounds = [transform.to_search(x) for x in self.support()]
            return Uniform(lower=np.min(bounds), upper=np.max(bounds))
        return None


class LogUniform(Distribution):
    """
    Represents a log-uniform distribution over a specified interval.

    Parameters
    ----------
    lower : float
        The lower bound of the distribution.
    upper : float
        The upper bound of the distribution.
    """

    def __init__(self, lower: float, upper: float):
        # Validate that upper > lower for all elements
        if not np.all(0 < lower < upper):
            raise ValueError(
                "All bounds must be positive and all elements of upper bounds "
                "must be greater than lower bounds."
            )
        properties = {"a": float(lower), "b": float(upper)}
        super().__init__(stats.loguniform(**properties), properties=properties)

    def __repr__(self) -> str:
        return f"{self.name}, bounds: {self.support()}"

    def _transform(self, transform: Transformation) -> BaseDistribution | None:
        """Get the transformed distribution in the search space."""
        if isinstance(transform, ScaledTransformation):
            bounds = [transform.to_search(x) for x in self.support()]
            return LogUniform(lower=np.min(bounds), upper=np.max(bounds))

        elif isinstance(transform, LogTransformation):
            bounds = [transform.to_search(x) for x in self.support()]
            return Uniform(lower=np.min(bounds), upper=np.max(bounds))
        return None


class Unbounded(BaseDistribution):
    """
    Represents an unbounded distribution with either zero or one finite bound.

    Parameters
    ----------
    lower : float
        The lower bound of the distribution.
    upper : float
        The upper bound of the distribution.
    """

    def __init__(
        self, initial_value: float = 1.0, lower: float = -np.inf, upper: float = np.inf
    ):
        super().__init__()
        self.lower = float(lower)
        self.upper = float(upper)
        self.initial_value = (
            None
            if initial_value is None
            else float(np.minimum(np.maximum(initial_value, lower), upper))
        )

    def support(self) -> tuple[float]:
        return (self.lower, self.upper)

    def _dlogpdf_dx(self, x):
        """Evaluates the first derivative of the log uniform distribution at x."""
        return np.zeros_like(x)

    def mean(self) -> float:
        """Returns an estimate for the mean of the distribution."""
        return self.initial_value

    def std(self) -> float:
        """Returns an estimate for the standard deivation of the distribution."""
        return 0.05 if self.initial_value == 0 else 0.05 * self.initial_value

    def __repr__(self) -> str:
        return f"{self.name}, initial value: {self.initial_value}, bounds: {self.support()}"

    def _transform(self, transform: Transformation) -> BaseDistribution | None:
        """Get the transformed distribution in the search space."""
        if isinstance(transform, ScaledTransformation | LogTransformation):
            bounds = [transform.to_search(x) for x in self.support()]
            return Unbounded(
                initial_value=transform.to_search(self.initial_value),
                lower=np.min(bounds),
                upper=np.max(bounds),
            )
        return None


class Exponential(Distribution):
    """
    Represents an exponential distribution with a specified scale parameter.

    Parameters
    ----------
    scale : float
        The scale parameter (lambda) of the exponential distribution.
    """

    def __init__(self, scale: float, loc: float = 0):
        properties = {"loc": float(loc), "scale": float(scale)}
        super().__init__(stats.expon(**properties), properties=properties)

    def _dlogpdf_dx(self, x):
        """Evaluates the first derivative of the log exponential distribution at x."""
        return -1 / self.properties["scale"] * np.ones_like(x)


class JointDistribution(BaseDistribution):
    """
    Represents a joint distribution composed of multiple distributions.

    Parameters
    ----------
    distributions : BaseDistribution
        One or more distributions to combine into a joint distribution.
    """

    def __init__(self, *distributions: BaseDistribution):
        super().__init__()

        if all(distribution is None for distribution in distributions):
            return

        for distribution in distributions:
            if not isinstance(distribution, BaseDistribution):
                raise ValueError(
                    "All distributions must be instances of BaseDistribution. "
                    f"Received {distribution}"
                )

        self._n_parameters = len(distributions)
        self._distributions_list: list[BaseDistribution] = list(distributions)

    def support(self) -> np.ndarray:
        """Return the support as a numpy array of dimensions (2, n_parameters)."""
        return np.asarray([d.support() for d in self._distributions_list]).T

    def mean(self):
        return np.asarray([d.mean() for d in self._distributions_list]).T

    def std(self):
        return np.asarray([d.std() for d in self._distributions_list])

    def cov(self):
        return np.diag(self.std()) ** 2

    def rvs(self, size: int = 1, random_state: int | None = None):
        """Sample each distribution individually and then compile."""
        all_samples = []
        for distribution in self._distributions_list:
            samples = distribution.rvs(size=size, random_state=random_state)
            all_samples.append(samples)
        return np.column_stack(all_samples)

    def logpdf(self, x: float | np.ndarray) -> float:
        """
        Evaluates the log of the joint distribution at a given point.

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
            d.logpdf(x) for d, x in zip(self._distributions_list, x, strict=False)
        )

    def logpdfS1(self, x: float | np.ndarray) -> tuple[float, np.ndarray]:
        """
        Evaluates the first derivative of the log of the joint distribution at a given point.

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

        for distribution, xi in zip(self._distributions_list, x, strict=False):
            p, dp = distribution.logpdfS1(xi)
            log_probs.append(p)
            derivatives.append(dp)

        output = sum(log_probs)
        doutput = np.asarray(derivatives)

        if doutput.ndim == 1:
            return output, doutput

        return output, doutput.T

    def __repr__(self) -> str:
        distributions_repr = "; ".join([repr(d) for d in self._distributions_list])
        return f"{self.name}(distributions: [{distributions_repr}])"

    def marginal(self, position: int):
        return self._distributions_list[position]

    def _transform(self, transform: Transformation) -> BaseDistribution | None:
        """Get the transformed distribution in the search space."""
        list_of_transforms = []
        for t, d in zip(
            transform.transformations, self._distributions_list, strict=False
        ):
            transformed_dist = d.get_transformed_distribution(t)
            if transformed_dist is None:
                return None
            else:
                list_of_transforms.append(transformed_dist)

        return JointDistribution(*list_of_transforms)
