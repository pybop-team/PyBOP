import pybop
import math


def plot_parameters(
    optim, xaxis_titles="Iteration", yaxis_titles=None, title="Convergence"
):
    """
    Plot the evolution of parameters during the optimization process using Plotly.

    Parameters
    ----------
    optim : object
        The optimization object containing the history of parameter values and associated cost.
    xaxis_titles : str, optional
        Title for the x-axis, defaulting to "Iteration".
    yaxis_titles : list of str, optional
        Titles for the y-axes, one for each parameter. If None, parameter names are used.
    title : str, optional
        Title of the plot, defaulting to "Convergence".

    Returns
    -------
    plotly.graph_objs.Figure
        A Plotly figure object showing the parameter evolution over iterations.
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
    Create traces for plotting parameter evolution.

    Parameters
    ----------
    params : list
        List of parameter objects, each having a 'name' attribute used for labeling the trace.
    trace_data : list of numpy.ndarray
        A list of arrays representing the historical values of each parameter.
    x_values : list or numpy.ndarray, optional
        The x-axis values for plotting. If None, defaults to sequential integers.

    Returns
    -------
    list of plotly.graph_objs.Scatter
        A list of Scatter trace objects, one for each parameter.
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
    Create a subplot with individual traces for each parameter.

    Parameters
    ----------
    traces : list of plotly.graph_objs.Scatter
        Traces to be plotted, one trace per subplot.
    plot_size : tuple of int, optional
        The size of the plot as (width, height), defaulting to (1024, 576).
    title : str, optional
        The title of the plot, defaulting to "Parameter Convergence".
    axis_titles : list of tuple of str, optional
        A list of (x_title, y_title) pairs for each subplot. If None, titles are omitted.
    **layout_kwargs : dict
        Additional keyword arguments to customize the layout.

    Returns
    -------
    plotly.graph_objs.Figure
        A Plotly figure object with subplots for each trace.
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
