from pybop import GaussianLogLikelihood, StandardSubplot


def plot_parameters(optim, show=True, **layout_kwargs):
    """
    Plot the evolution of parameters during the optimization process using Plotly.

    Parameters
    ----------
    optim : object
        Optimisation object containing the history of parameter values and associated cost.
    show : bool, optional
        If True, the figure is shown upon creation (default: True).
    **layout_kwargs : optional
        Valid Plotly layout keys and their values,
        e.g. `xaxis_title="Time [s]"` or
        `xaxis={"title": "Time [s]", "titlefont_size": 18}`.

    Returns
    -------
    plotly.graph_objs.Figure
        A Plotly figure object showing the parameter evolution over iterations.
    """

    # Extract parameters and log from the optimisation object
    parameters = optim.cost.parameters
    log = optim.log["x"]

    # Create a list of sequential integers for the x-axis
    x = list(range(1, len(log[0]) * len(log) + 1))

    # Determine the number of elements in the smallest arrays
    num_elements = len(log[0][0])

    # Initialize a list of lists to store our columns
    y = [[] for _ in range(num_elements)]

    # Loop through each numpy array in y
    for array in log:
        # Loop through each item (which is a n-element array) in the numpy array
        for item in array:
            # Loop through each element in the item and append to the corresponding column
            for i in range(num_elements):
                y[i].append(item[i])

    # Create lists of axis titles and trace names
    axis_titles = []
    trace_names = parameters.keys()
    for name in trace_names:
        axis_titles.append(("Function Call", name))

    if isinstance(optim.cost, GaussianLogLikelihood):
        axis_titles.append(("Function Call", "Sigma"))
        trace_names.append("Sigma")

    # Set subplot layout options
    layout_options = dict(
        title="Parameter Convergence",
        width=1024,
        height=576,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # Create a plotting dictionary
    plot_dict = StandardSubplot(
        x=x,
        y=y,
        axis_titles=axis_titles,
        layout_options=layout_options,
        trace_names=trace_names,
        trace_name_width=50,
    )

    # Generate the figure and update the layout
    fig = plot_dict(show=False)
    fig.update_layout(**layout_kwargs)
    if show:
        fig.show()

    return fig
