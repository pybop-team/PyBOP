import pybop
import numpy as np
import scipy.stats as stats


class Gaussian:
    """
    Gaussian prior class.
    """

    def __init__(self, mean, sigma):
        self.name = "Gaussian"
        self.mean = mean
        self.sigma = sigma

    def pdf(self, x):
        return stats.norm.pdf(x, loc=self.mean, scale=self.sigma)

    def logpdf(self, x):
        return stats.norm.logpdf(x, loc=self.mean, scale=self.sigma)

    def rvs(self, size):
        if size < 0:
            raise ValueError("size must be positive")
        else:
            return stats.norm.rvs(loc=self.mean, scale=self.sigma, size=size)
    
    def __repr__(self):
        return f"{self.name}, mean: {self.mean}, sigma: {self.sigma}"

class Uniform:
    """
    Uniform prior class.
    """

    def __init__(self, lower, upper):
        self.name = "Uniform"
        self.lower = lower
        self.upper = upper

    def pdf(self, x):
        return stats.uniform.pdf(x, loc=self.lower, scale=self.upper - self.lower)

    def logpdf(self, x):
        return stats.uniform.logpdf(x, loc=self.lower, scale=self.upper - self.lower)

    def rvs(self, size):
        if size < 0:
            raise ValueError("size must be positive")
        else:
            return stats.uniform.rvs(loc=self.lower, scale=self.upper - self.lower, size=size)
    def __repr__(self):
        return f"{self.name}, lower: {self.lower}, upper: {self.upper}"

class Exponential:
    """
    exponential prior class.
    """

    def __init__(self, scale):
        self.name = "Exponential"
        self.scale = scale

    def pdf(self, x):
        return stats.expon.pdf(x, scale=self.scale)

    def logpdf(self, x):
        return stats.expon.logpdf(x, scale=self.scale)

    def rvs(self, size):
        if size < 0:
            raise ValueError("size must be positive")
        else:
            return stats.expon.rvs(scale=self.scale, size=size)
    def __repr__(self):
        return f"{self.name}, scale: {self.scale}"