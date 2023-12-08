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


   Exponential prior class.

   .. py:method:: __repr__()

      Return repr(self).


   .. py:method:: logpdf(x)


   .. py:method:: pdf(x)


   .. py:method:: rvs(size)



.. py:class:: Gaussian(mean, sigma)


   Gaussian prior class.

   .. py:method:: __repr__()

      Return repr(self).


   .. py:method:: logpdf(x)


   .. py:method:: pdf(x)


   .. py:method:: rvs(size)



.. py:class:: Uniform(lower, upper)


   Uniform prior class.

   .. py:method:: __repr__()

      Return repr(self).


   .. py:method:: logpdf(x)


   .. py:method:: pdf(x)


   .. py:method:: rvs(size)
