import pybop
import math


def plot_parameters(
    optim, xaxis_titles="Iteration", yaxis_titles=None, title="Convergence"
):
    """
    Plot the evolution of the parameters during the optimisation process.

    Parameters:
    ------------
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
    ----------
    fig : plotly.graph_objs.Figure
        The Plotly figure object for the plot depicting how the parameters of
        the optimisation algorithm evolve over its course. This can be useful
        for diagnosing the behaviour of the optimisation algorithm.

    Notes:
    ----------
    The function assumes that the 'optim' object has a 'cost.problem.parameters'
    attribute containing the parameters of the optimisation algorithm and a 'log'
    attribute containing a history of the iterations.
    """

    # Extract parameters from the optimisation object
    params = optim.cost.problem.parameters

    # Create the traces from the optimisation log
    traces = create_traces(params, optim.log)

    # Create the axis titles
    axis_titles = []
    for param in params:
        axis_titles.append(("Function Call", param.name))

    # Create the figure
    fig = create_subplots_with_traces(traces, axis_titles=axis_titles)

    # Display the figure
    fig.show()

    return fig


def create_traces(params, trace_data, x_values=None):
    """
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
    """

    # Attempt to import plotly when an instance is created
    go = pybop.PlotlyManager().go

    traces = []

    # If x_values are not provided:
    if x_values is None:
        x_values = list(range(len(trace_data[0]) * len(trace_data)))

    # Determine the number of elements in the smallest arrays
    num_elements = len(trace_data[0][0])

    # Initialize a list of lists to store our columns
    columns = [[] for _ in range(num_elements)]

    # Loop through each numpy array in trace_data
    for array in trace_data:
        # Loop through each item (which is a n-element array) in the numpy array
        for item in array:
            # Loop through each element in the item and append to the corresponding column
            for i in range(num_elements):
                columns[i].append(item[i])

    # Create a trace for each column
    for i in range(len(columns)):
        wrap_param = pybop.StandardPlot.wrap_text(params[i].name, width=50)
        traces.append(go.Scatter(x=x_values, y=columns[i], name=wrap_param))

    return traces


def create_subplots_with_traces(
    traces,
    plot_size=(1024, 576),
    title="Parameter Convergence",
    axis_titles=None,
    **layout_kwargs,
):
    """
    Creates a subplot figure with the given traces.

    :param traces: List of plotly.graph_objs traces that will be added to the subplots.
    :param plot_size: Tuple (width, height) representing the desired size of the plot.
    :param title: The main title of the subplot figure.
    :param axis_titles: List of tuples for axis titles in the form [(x_title, y_title), ...] for each subplot.
    :param layout_kwargs: Additional keyword arguments to be passed to fig.update_layout for custom layout.
    :return: A plotly figure object with the subplots.
    """

    # Attempt to import plotly when an instance is created
    make_subplots = pybop.PlotlyManager().make_subplots

    num_traces = len(traces)
    num_cols = int(math.ceil(math.sqrt(num_traces)))
    num_rows = int(math.ceil(num_traces / num_cols))

    fig = make_subplots(rows=num_rows, cols=num_cols, start_cell="bottom-left")

    for idx, trace in enumerate(traces):
        row = (idx // num_cols) + 1
        col = (idx % num_cols) + 1
        fig.add_trace(trace, row=row, col=col)

        if axis_titles and idx < len(axis_titles):
            x_title, y_title = axis_titles[idx]
            fig.update_xaxes(title_text=x_title, row=row, col=col)
            fig.update_yaxes(title_text=y_title, row=row, col=col)

    if plot_size:
        layout_kwargs["width"], layout_kwargs["height"] = plot_size

    if title:
        layout_kwargs["title_text"] = title

    # Set the legend above the subplots
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **layout_kwargs,
    )

    return fig
