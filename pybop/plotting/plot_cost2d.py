import numpy as np


def plot_cost2d(cost, bounds=None, optim=None, steps=10):
    """
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
    costs = np.zeros((len(x), len(y)))

    # Populate cost matrix
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            costs[i, j] = cost([xi, yj])

    # Create figure
    fig = create_figure(x, y, costs, bounds, cost.problem.parameters, optim)

    # Display figure
    fig.show()

    return fig


def get_param_bounds(cost):
    """
    Use parameters bounds for range of cost landscape
    """
    bounds = np.empty((len(cost.problem.parameters), 2))
    for i, param in enumerate(cost.problem.parameters):
        bounds[i] = param.bounds
    return bounds


def create_figure(x, y, z, bounds, params, optim):
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
                x=optim_trace[0:-1, 0],
                y=optim_trace[0:-1, 1],
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
