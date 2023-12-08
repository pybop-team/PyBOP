:py:mod:`pybop.plotting.plot_parameters`
========================================

.. py:module:: pybop.plotting.plot_parameters


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   pybop.plotting.plot_parameters.create_subplots_with_traces
   pybop.plotting.plot_parameters.create_traces
   pybop.plotting.plot_parameters.plot_parameters



.. py:function:: create_subplots_with_traces(traces, plot_size=(1024, 576), title='Parameter Convergence', axis_titles=None, **layout_kwargs)

   Creates a subplot figure with the given traces.

   :param traces: List of plotly.graph_objs traces that will be added to the subplots.
   :param plot_size: Tuple (width, height) representing the desired size of the plot.
   :param title: The main title of the subplot figure.
   :param axis_titles: List of tuples for axis titles in the form [(x_title, y_title), ...] for each subplot.
   :param layout_kwargs: Additional keyword arguments to be passed to fig.update_layout for custom layout.
   :return: A plotly figure object with the subplots.


.. py:function:: create_traces(params, trace_data, x_values=None)

   Generate a list of Plotly Scatter trace objects from provided trace data.

   This function assumes that each column in the ``trace_data`` represents a separate trace to be plotted,
   and that the ``params`` list contains objects with a ``name`` attribute used for trace names.
   Text wrapping for trace names is performed by ``pybop.StandardPlot.wrap_text``.

   Parameters:
   - params (list): A list of objects, where each object has a ``name`` attribute used as the trace name.
                    The list should have the same length as the number of traces in ``trace_data``.
   - trace_data (list of lists): A 2D list where each inner list represents y-values for a trace.
   - x_values (list, optional): A list of x-values to be used for all traces. If not provided, a
                                range of integers starting from 0 will be used.

   Returns:
   - list: A list of Plotly ``go.Scatter`` objects, each representing a trace to be plotted.

   Notes:
   - The function depends on ``pybop.StandardPlot.wrap_text`` for text wrapping, which needs to be available
     in the execution context.
   - The function assumes that ``go`` from ``plotly.graph_objs`` is already imported as ``go``.


.. py:function:: plot_parameters(optim, xaxis_titles='Iteration', yaxis_titles=None, title='Convergence')

   Plot the evolution of the parameters during the optimisation process.

   Parameters:
   ----------
   optim : optimisation object
       An object representing the optimisation process, which should contain
       information about the cost function, optimiser, and the history of the
       parameter values throughout the iterations.
   xaxis_title : str, optional
       Title for the x-axis, representing the iteration number or a similar
       discrete time step in the optimisation process (default is "Iteration").
   yaxis_title : str, optional
       Title for the y-axis, which typically represents the metric being
       optimised, such as cost or loss (default is "Cost").
   title : str, optional
       Title of the plot, which provides an overall description of what the
       plot represents (default is "Convergence").

   Returns:
   -------
   fig : plotly.graph_objs.Figure
       The Plotly figure object for the plot depicting how the parameters of
       the optimisation algorithm evolve over its course. This can be useful
       for diagnosing the behaviour of the optimisation algorithm.

   Notes:
   -----
   The function assumes that the 'optim' object has a 'cost.problem.parameters'
   attribute containing the parameters of the optimisation algorithm and a 'log'
   attribute containing a history of the iterations.
