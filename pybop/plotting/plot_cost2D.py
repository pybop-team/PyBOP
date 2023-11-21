import numpy as np


def plot_cost2D(cost, steps=10):
    """
    Query the cost landscape for a given parameter space and plot using plotly.
    """

    # Set up parameter bounds
    bounds = get_param_bounds(cost)

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
    fig = create_figure(x, y, costs, bounds, cost.problem.parameters)

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


def create_figure(x, y, z, bounds, params):
    # Import plotly only when needed
    import plotly.graph_objects as go

    fig = go.Figure(data=[go.Contour(x=x, y=y, z=z)])
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
