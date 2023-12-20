import numpy as np


def plot_cost2d(cost, bounds=None, optim=None, steps=10):
    """
    Plot a 2D visualization of a cost landscape using Plotly.

    This function generates a contour plot representing the cost landscape for a provided
    callable cost function over a grid of parameter values within the specified bounds.

    Parameters
    ----------
    cost : callable
        The cost function to be evaluated. Must accept a list of parameters and return a cost value.
    bounds : numpy.ndarray, optional
        A 2x2 array specifying the [min, max] bounds for each parameter. If None, uses `get_param_bounds`.
    optim : object, optional
        An optimiser instance which, if provided, overlays its specific trace on the plot.
    steps : int, optional
        The number of intervals to divide the parameter space into along each dimension (default is 10).

    Returns
    -------
    plotly.graph_objs.Figure
        The Plotly figure object containing the cost landscape plot.

    Raises
    ------
    ValueError
        If the cost function does not return a valid cost when called with a parameter list.
    """

    if bounds is None:
        # Set up parameter bounds
        bounds = get_param_bounds(cost)
    else:
        bounds = bounds

    # Generate grid
    x = np.linspace(bounds[0, 0], bounds[0, 1], steps)
    y = np.linspace(bounds[1, 0], bounds[1, 1], steps)

    # Initialize cost matrix
    costs = np.zeros((len(y), len(x)))

    # Populate cost matrix
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            costs[j, i] = cost([xi, yj])

    # Create figure
    fig = create_figure(x, y, costs, bounds, cost.problem.parameters, optim)

    # Display figure
    fig.show()

    return fig


def get_param_bounds(cost):
    """
    Retrieve parameter bounds from a cost function's associated problem parameters.

    Parameters
    ----------
    cost : callable
        The cost function with an associated 'problem' attribute containing 'parameters'.

    Returns
    -------
    numpy.ndarray
        An array of shape (n_parameters, 2) containing the bounds for each parameter.
    """
    bounds = np.empty((len(cost.problem.parameters), 2))
    for i, param in enumerate(cost.problem.parameters):
        bounds[i] = param.bounds
    return bounds


def create_figure(x, y, z, bounds, params, optim):
    """
    Create a Plotly figure with a 2D contour plot of the cost landscape.

    Parameters
    ----------
    x : numpy.ndarray
        1D array of x-coordinates for the meshgrid.
    y : numpy.ndarray
        1D array of y-coordinates for the meshgrid.
    z : numpy.ndarray
        2D array of cost function values corresponding to the meshgrid.
    bounds : numpy.ndarray
        A 2x2 array specifying the [min, max] bounds for each parameter.
    params : iterable
        An iterable of parameter objects with 'name' attributes for axis labeling.
    optim : object
        An optimiser instance with 'log' and 'x0' attributes for plotting traces.

    Returns
    -------
    plotly.graph_objs.Figure
        The Plotly figure object with the contour plot and optimization traces.
    """

    # Import plotly only when needed
    import plotly.graph_objects as go

    fig = go.Figure(data=[go.Contour(x=x, y=y, z=z)])
    if optim is not None:
        optim_trace = np.array([item for sublist in optim.log for item in sublist])
        optim_trace = optim_trace.reshape(-1, 2)

        # Plot initial guess
        fig.add_trace(
            go.Scatter(
                x=[optim.x0[0]],
                y=[optim.x0[1]],
                mode="markers",
                marker_symbol="x",
                marker=dict(
                    color="red",
                    line_color="midnightblue",
                    line_width=1,
                    size=12,
                    showscale=False,
                ),
                showlegend=False,
            )
        )

        # Plot optimisation trace
        fig.add_trace(
            go.Scatter(
                x=optim_trace[:, 0],
                y=optim_trace[:, 1],
                mode="markers",
                marker=dict(
                    color=[i / len(optim_trace) for i in range(len(optim_trace))],
                    colorscale="YlOrBr",
                    showscale=False,
                ),
                showlegend=False,
            )
        )

    # Set figure properties
    fig.update_layout(
        title="Cost Landscape",
        title_x=0.5,
        title_y=0.9,
        xaxis_title=params[0].name,
        yaxis_title=params[1].name,
        width=600,
        height=600,
        xaxis=dict(range=bounds[0]),
        yaxis=dict(range=bounds[1]),
    )

    return fig
