:py:mod:`pybop.plotting.plot_cost2d`
====================================

.. py:module:: pybop.plotting.plot_cost2d


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   pybop.plotting.plot_cost2d.create_figure
   pybop.plotting.plot_cost2d.get_param_bounds
   pybop.plotting.plot_cost2d.plot_cost2d



.. py:function:: create_figure(x, y, z, bounds, params, optim)


.. py:function:: get_param_bounds(cost)

   Use parameters bounds for range of cost landscape


.. py:function:: plot_cost2d(cost, bounds=None, optim=None, steps=10)

   Query the cost landscape for a given parameter space and plot using plotly.
