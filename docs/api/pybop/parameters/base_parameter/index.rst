:py:mod:`pybop.parameters.base_parameter`
=========================================

.. py:module:: pybop.parameters.base_parameter


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pybop.parameters.base_parameter.Parameter




.. py:class:: Parameter(name, value=None, prior=None, bounds=None)


   ""
   Class for creating parameters in PyBOP.

   .. py:method:: __repr__()

      Return repr(self).


   .. py:method:: rvs(n_samples)

      Returns a random value sample from the prior distribution.


   .. py:method:: set_margin(margin)

      Sets the margin for the parameter.


   .. py:method:: update(value)
