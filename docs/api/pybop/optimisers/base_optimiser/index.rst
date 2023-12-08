:py:mod:`pybop.optimisers.base_optimiser`
=========================================

.. py:module:: pybop.optimisers.base_optimiser


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pybop.optimisers.base_optimiser.BaseOptimiser




.. py:class:: BaseOptimiser


   Base class for the optimisation methods.


   .. py:method:: _runoptimise(cost_function, x0=None, bounds=None)

      Run optimisation method, to be overloaded by child classes.



   .. py:method:: name()

      Returns the name of the optimiser.


   .. py:method:: optimise(cost_function, x0=None, bounds=None)

      Optimisiation method to be overloaded by child classes.
