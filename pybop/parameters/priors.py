import scipy.stats as stats


class Gaussian:
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

    def __init__(self, mean, sigma):
        self.name = "Gaussian"
        self.mean = mean
        self.sigma = sigma

    def pdf(self, x):
        """
        Calculates the probability density function of the Gaussian distribution at x.

        Parameters
        ----------
        x : float
            The point at which to evaluate the pdf.

        Returns
        -------
        float
            The probability density function value at x.
        """
        return stats.norm.pdf(x, loc=self.mean, scale=self.sigma)

    def logpdf(self, x):
        """
        Calculates the logarithm of the probability density function of the Gaussian distribution at x.

        Parameters
        ----------
        x : float
            The point at which to evaluate the log pdf.

        Returns
        -------
        float
            The logarithm of the probability density function value at x.
        """
        return stats.norm.logpdf(x, loc=self.mean, scale=self.sigma)

    def rvs(self, size):
        """
        Generates random variates from the Gaussian distribution.

        Parameters
        ----------
        size : int
            The number of random variates to generate.

        Returns
        -------
        array_like
            An array of random variates from the Gaussian distribution.

        Raises
        ------
        ValueError
            If the size parameter is negative.
        """
        if size < 0:
            raise ValueError("size must be positive")
        else:
            return stats.norm.rvs(loc=self.mean, scale=self.sigma, size=size)

    def __repr__(self):
        """
        Returns a string representation of the Gaussian object.
        """
        return f"{self.name}, mean: {self.mean}, sigma: {self.sigma}"


class Uniform:
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

    def __init__(self, lower, upper):
        self.name = "Uniform"
        self.lower = lower
        self.upper = upper

    def pdf(self, x):
        """
        Calculates the probability density function of the uniform distribution at x.

        Parameters
        ----------
        x : float
            The point at which to evaluate the pdf.

        Returns
        -------
        float
            The probability density function value at x.
        """
        return stats.uniform.pdf(x, loc=self.lower, scale=self.upper - self.lower)

    def logpdf(self, x):
        """
        Calculates the logarithm of the pdf of the uniform distribution at x.

        Parameters
        ----------
        x : float
            The point at which to evaluate the log pdf.

        Returns
        -------
        float
            The log of the probability density function value at x.
        """
        return stats.uniform.logpdf(x, loc=self.lower, scale=self.upper - self.lower)

    def rvs(self, size):
        """
        Generates random variates from the uniform distribution.

        Parameters
        ----------
        size : int
            The number of random variates to generate.

        Returns
        -------
        array_like
            An array of random variates from the uniform distribution.

        Raises
        ------
        ValueError
            If the size parameter is negative.
        """
        if size < 0:
            raise ValueError("size must be positive")
        else:
            return stats.uniform.rvs(
                loc=self.lower, scale=self.upper - self.lower, size=size
            )

    def __repr__(self):
        """
        Returns a string representation of the Uniform object.
        """
        return f"{self.name}, lower: {self.lower}, upper: {self.upper}"


class Exponential:
    """
    Represents an exponential distribution with a specified scale parameter.

    This class provides methods to calculate the pdf, the log pdf, and to generate random
    variates from the distribution.

    Parameters
    ----------
    scale : float
        The scale parameter (lambda) of the exponential distribution.
    """

    def __init__(self, scale):
        self.name = "Exponential"
        self.scale = scale

    def pdf(self, x):
        """
        Calculates the probability density function of the exponential distribution at x.

        Parameters
        ----------
        x : float
            The point at which to evaluate the pdf.

        Returns
        -------
        float
            The probability density function value at x.
        """
        return stats.expon.pdf(x, scale=self.scale)

    def logpdf(self, x):
        """
        Calculates the logarithm of the pdf of the exponential distribution at x.

        Parameters
        ----------
        x : float
            The point at which to evaluate the log pdf.

        Returns
        -------
        float
            The log of the probability density function value at x.
        """
        return stats.expon.logpdf(x, scale=self.scale)

    def rvs(self, size):
        """
        Generates random variates from the exponential distribution.

        Parameters
        ----------
        size : int
            The number of random variates to generate.

        Returns
        -------
        array_like
            An array of random variates from the exponential distribution.

        Raises
        ------
        ValueError
            If the size parameter is negative.
        """
        if size < 0:
            raise ValueError("size must be positive")
        else:
            return stats.expon.rvs(scale=self.scale, size=size)

    def __repr__(self):
        """
        Returns a string representation of the Uniform object.
        """
        return f"{self.name}, scale: {self.scale}"
