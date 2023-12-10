:py:mod:`pybop.parameters.priors`
=================================

.. py:module:: pybop.parameters.priors


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pybop.parameters.priors.Exponential
   pybop.parameters.priors.Gaussian
   pybop.parameters.priors.Uniform




.. py:class:: Exponential(scale)


   Represents an exponential distribution with a specified scale parameter.

   This class provides methods to calculate the pdf, the log pdf, and to generate random
   variates from the distribution.

   :param scale: The scale parameter (lambda) of the exponential distribution.
   :type scale: float

   .. py:method:: __repr__()

      Returns a string representation of the Uniform object.


   .. py:method:: logpdf(x)

      Calculates the logarithm of the pdf of the exponential distribution at x.

      :param x: The point at which to evaluate the log pdf.
      :type x: float

      :returns: The log of the probability density function value at x.
      :rtype: float


   .. py:method:: pdf(x)

      Calculates the probability density function of the exponential distribution at x.

      :param x: The point at which to evaluate the pdf.
      :type x: float

      :returns: The probability density function value at x.
      :rtype: float


   .. py:method:: rvs(size)

      Generates random variates from the exponential distribution.

      :param size: The number of random variates to generate.
      :type size: int

      :returns: An array of random variates from the exponential distribution.
      :rtype: array_like

      :raises ValueError: If the size parameter is not positive.



.. py:class:: Gaussian(mean, sigma)


   Represents a Gaussian (normal) distribution with a given mean and standard deviation.

   This class provides methods to calculate the probability density function (pdf),
   the logarithm of the pdf, and to generate random variates (rvs) from the distribution.

   :param mean: The mean (mu) of the Gaussian distribution.
   :type mean: float
   :param sigma: The standard deviation (sigma) of the Gaussian distribution.
   :type sigma: float

   .. py:method:: __repr__()

      Returns a string representation of the Gaussian object.


   .. py:method:: logpdf(x)

      Calculates the logarithm of the probability density function of the Gaussian distribution at x.

      :param x: The point at which to evaluate the log pdf.
      :type x: float

      :returns: The logarithm of the probability density function value at x.
      :rtype: float


   .. py:method:: pdf(x)

      Calculates the probability density function of the Gaussian distribution at x.

      :param x: The point at which to evaluate the pdf.
      :type x: float

      :returns: The probability density function value at x.
      :rtype: float


   .. py:method:: rvs(size)

      Generates random variates from the Gaussian distribution.

      :param size: The number of random variates to generate.
      :type size: int

      :returns: An array of random variates from the Gaussian distribution.
      :rtype: array_like

      :raises ValueError: If the size parameter is not positive.



.. py:class:: Uniform(lower, upper)


   Represents a uniform distribution over a specified interval.

   This class provides methods to calculate the pdf, the log pdf, and to generate
   random variates from the distribution.

   :param lower: The lower bound of the distribution.
   :type lower: float
   :param upper: The upper bound of the distribution.
   :type upper: float

   .. py:method:: __repr__()

      Returns a string representation of the Uniform object.


   .. py:method:: logpdf(x)

      Calculates the logarithm of the pdf of the uniform distribution at x.

      :param x: The point at which to evaluate the log pdf.
      :type x: float

      :returns: The log of the probability density function value at x.
      :rtype: float


   .. py:method:: pdf(x)

      Calculates the probability density function of the uniform distribution at x.

      :param x: The point at which to evaluate the pdf.
      :type x: float

      :returns: The probability density function value at x.
      :rtype: float


   .. py:method:: rvs(size)

      Generates random variates from the uniform distribution.

      :param size: The number of random variates to generate.
      :type size: int

      :returns: An array of random variates from the uniform distribution.
      :rtype: array_like

      :raises ValueError: If the size parameter is not positive.
