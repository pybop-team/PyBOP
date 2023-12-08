:py:mod:`pybop.costs.standalone`
================================

.. py:module:: pybop.costs.standalone


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pybop.costs.standalone.StandaloneCost




.. py:class:: StandaloneCost(problem=None)


   Bases: :py:obj:`pybop.BaseCost`

   Base class for defining cost functions.
   This class computes a corresponding goodness-of-fit for a corresponding model prediction and dataset.
   Lower cost values indicate a better fit.

   .. py:method:: __call__(x, grad=None)

      Returns the cost function value and computes the cost.
