import pybop
import numpy as np


def plot_cost2d(cost, bounds=None, steps=10, show=True, **layout_kwargs):
    """
    Plot a 2D visualisation of a cost landscape using Plotly.

    This function generates a contour plot representing the cost landscape for a provided
    callable cost function over a grid of parameter values within the specified bounds.

    Parameters
    ----------
    cost : callable
        The cost function to be evaluated. Must accept a list of parameter values and return a cost value.
    bounds : numpy.ndarray, optional
        A 2x2 array specifying the [min, max] bounds for each parameter. If None, uses `get_param_bounds`.
    steps : int, optional
        The number of intervals to divide the parameter space into along each dimension (default is 10).
    show : bool, optional
        If True, the figure is shown upon creation (default: True).
    **layout_kwargs : optional
        Valid Plotly layout keys and their values,
        e.g. `xaxis_title="Time [s]"` or
        `xaxis={"title": "Time [s]", "titlefont_size": 18}`.

    Returns
    -------
    plotly.graph_objs.Figure
        The Plotly figure object containing the cost landscape plot.

    Raises
    ------
    ValueError
        If the cost function does not return a valid cost when called with a parameter list.
    """

    # Set up parameter bounds
    if bounds is None:
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
            costs[j, i] = cost(np.array([xi, yj]))

    # Import plotly only when needed
    go = pybop.PlotlyManager().go

    # Set default layout properties
    layout_options = dict(
        title="Cost Landscape",
        title_x=0.5,
        title_y=0.9,
        width=600,
        height=600,
        xaxis=dict(range=bounds[0]),
        yaxis=dict(range=bounds[1]),
    )
    if hasattr(cost, "parameters"):
        layout_options["xaxis_title"] = cost.parameters[0].name
        layout_options["yaxis_title"] = cost.parameters[1].name
    layout = go.Layout(layout_options)

    # Create contour plot and update the layout
    fig = go.Figure(data=[go.Contour(x=x, y=y, z=costs)], layout=layout)
    fig.update_layout(**layout_kwargs)
    if show:
        fig.show()

    return fig


def get_param_bounds(cost):
    """
    Retrieve parameter bounds from a cost object's associated problem parameters.

    Parameters
    ----------
    cost : object
        The cost object with an associated 'problem' attribute containing 'parameters'.

    Returns
    -------
    bounds : numpy.ndarray
        An array of shape (n_parameters, 2) containing the bounds for each parameter.
    """
    if hasattr(cost, "problem"):
        bounds = np.empty((len(cost.parameters), 2))
        for i, param in enumerate(cost.parameters):
            bounds[i] = param.bounds
        return bounds

    return None
