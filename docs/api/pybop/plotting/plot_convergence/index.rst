:py:mod:`pybop.plotting.plot_convergence`
=========================================

.. py:module:: pybop.plotting.plot_convergence


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   pybop.plotting.plot_convergence.plot_convergence



.. py:function:: plot_convergence(optim, xaxis_title='Iteration', yaxis_title='Cost', title='Convergence')

   Plot the convergence of the optimisation algorithm.

   Parameters:
   ----------
   optim : optimisation object
       Optimisation object containing the cost function and optimiser.
   xaxis_title : str, optional
       Title for the x-axis (default is "Iteration").
   yaxis_title : str, optional
       Title for the y-axis (default is "Cost").
   title : str, optional
       Title of the plot (default is "Convergence").

   Returns:
   ----------
   fig : plotly.graph_objs.Figure
       The Plotly figure object for the convergence plot.
