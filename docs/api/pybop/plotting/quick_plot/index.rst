:py:mod:`pybop.plotting.quick_plot`
===================================

.. py:module:: pybop.plotting.quick_plot


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pybop.plotting.quick_plot.StandardPlot



Functions
~~~~~~~~~

.. autoapisummary::

   pybop.plotting.quick_plot.quick_plot



.. py:class:: StandardPlot(x, y, cost, y2=None, title=None, xaxis_title=None, yaxis_title=None, trace_name=None, width=1024, height=576)


   A class for creating and displaying a plotly figure that compares a target dataset against a simulated model output.

   This class provides an interface for generating interactive plots using Plotly, with the ability to include an
   optional secondary dataset and visualize uncertainty if provided.

   Attributes:
   -----------
   x : list
       The x-axis data points.
   y : list or np.ndarray
       The primary y-axis data points representing the simulated model output.
   y2 : list or np.ndarray, optional
       An optional secondary y-axis data points representing the target dataset against which the model output is compared.
   cost : float
       The cost associated with the model output.
   title : str, optional
       The title of the plot.
   xaxis_title : str, optional
       The title for the x-axis.
   yaxis_title : str, optional
       The title for the y-axis.
   trace_name : str, optional
       The name of the primary trace representing the model output. Defaults to "Simulated".
   width : int, optional
       The width of the figure in pixels. Defaults to 720.
   height : int, optional
       The height of the figure in pixels. Defaults to 540.

   Example:
   ----------
   >>> x_data = [1, 2, 3, 4]
   >>> y_simulated = [10, 15, 13, 17]
   >>> y_target = [11, 14, 12, 16]
   >>> plot = pybop.StandardPlot(x_data, y_simulated, cost=0.05, y2=y_target,
                           title="Model vs. Target", xaxis_title="X Axis", yaxis_title="Y Axis")
   >>> fig = plot()  # Generate the figure
   >>> fig.show()    # Display the figure in a browser

   .. py:method:: __call__()

      Generate the plotly figure.


   .. py:method:: create_layout()

      Create the layout for the plot.


   .. py:method:: create_traces()

      Create the traces for the plot.


   .. py:method:: wrap_text(text, width)
      :staticmethod:

      Wrap text to a specified width.

      Parameters:
      -----------
      text: str
          Text to be wrapped.
      width: int
          Width to wrap text to.

      Returns:
      ----------
      str
          Wrapped text with HTML line breaks.



.. py:function:: quick_plot(params, cost, title='Scatter Plot', width=1024, height=576)

   Plot the target dataset against the minimised model output.

   Parameters:
   -----------
   params : array-like
       Optimised parameters.
   cost : cost object
       Cost object containing the problem, dataset, and signal.
   title : str, optional
       Title of the plot (default is "Scatter Plot").
   width : int, optional
       Width of the figure in pixels (default is 720).
   height : int, optional
       Height of the figure in pixels (default is 540).

   Returns:
   ----------
   fig : plotly.graph_objs.Figure
       The Plotly figure object for the scatter plot.
