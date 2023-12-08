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

   Query the cost landscape for a given parameter space and plot it using Plotly.

   This function creates a 2D plot that visualizes the cost landscape over a grid
   of points within specified parameter bounds. If no bounds are provided, it determines
   them from the bounds on the parameter class.

   :param cost: A callable representing the cost function to be queried. It should
                take a list of parameters and return a cost value.
   :type cost: callable
   :param bounds: The bounds for the parameter space as a 2x2 array, with each
                  sub-array representing the min and max bounds for a parameter.
                  If None, bounds will be determined by `get_param_bounds`.
   :type bounds: numpy.ndarray, optional
   :param optim: An optional optimizer instance. If provided, it will be used to
                 overlay optimizer-specific information on the plot.
   :type optim: object, optional
   :param steps: The number of steps to divide the parameter space grid. More steps
                 result in finer resolution but increase computational cost.
   :type steps: int, optional
   :return: A Plotly figure object representing the cost landscape plot.
   :rtype: plotly.graph_objs.Figure

   :raises ValueError: If the cost function does not behave as expected.
