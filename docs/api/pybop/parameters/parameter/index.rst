:py:mod:`pybop.parameters.parameter`
====================================

.. py:module:: pybop.parameters.parameter


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pybop.parameters.parameter.Parameter




.. py:class:: Parameter(name, initial_value=None, prior=None, bounds=None)


   Represents a parameter within the PyBOP framework.

   This class encapsulates the definition of a parameter, including its name, prior
   distribution, initial value, bounds, and a margin to ensure the parameter stays
   within feasible limits during optimization or sampling.

   :param name: The name of the parameter.
   :type name: str
   :param initial_value: The initial value to be assigned to the parameter. Defaults to None.
   :type initial_value: float, optional
   :param prior: The prior distribution from which parameter values are drawn. Defaults to None.
   :type prior: scipy.stats distribution, optional
   :param bounds: A tuple defining the lower and upper bounds for the parameter.
                  Defaults to None.
   :type bounds: tuple, optional

   .. method:: rvs(n_samples)

      Draw random samples from the parameter's prior distribution.

   .. method:: update(value)

      Update the parameter's current value.

   .. method:: set_margin(margin)

      Set the margin to a specified positive value less than 1.


   :raises ValueError: If the lower bound is not strictly less than the upper bound, or if
       the margin is set outside the interval (0, 1).

   .. py:method:: __repr__()

      Return a string representation of the Parameter instance.

      :returns: A string including the parameter's name, prior, bounds, and current value.
      :rtype: str


   .. py:method:: rvs(n_samples)

      Draw random samples from the parameter's prior distribution.

      The samples are constrained to be within the parameter's bounds, excluding
      a predefined margin at the boundaries.

      :param n_samples: The number of samples to draw.
      :type n_samples: int

      :returns: An array of samples drawn from the prior distribution within the parameter's bounds.
      :rtype: array-like


   .. py:method:: set_margin(margin)

      Set the margin to a specified positive value less than 1.

      The margin is used to ensure parameter samples are not drawn exactly at the bounds,
      which may be problematic in some optimization or sampling algorithms.

      :param margin: The new margin value to be used, which must be in the interval (0, 1).
      :type margin: float

      :raises ValueError: If the margin is not between 0 and 1.


   .. py:method:: update(value)

      Update the parameter's current value.

      :param value: The new value to be assigned to the parameter.
      :type value: float
