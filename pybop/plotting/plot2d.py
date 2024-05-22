import sys

import numpy as np

import pybop


def plot2d(
    cost_or_optim, gradient=False, bounds=None, steps=10, show=True, **layout_kwargs
):
    """
    Plot a 2D visualisation of a cost landscape using Plotly.

    This function generates a contour plot representing the cost landscape for a provided
    callable cost function over a grid of parameter values within the specified bounds.

    Parameters
    ----------
    cost_or_optim : a callable cost function, pybop Cost or Optimisation object
        Either:
        - the cost function to be evaluated. Must accept a list of parameter values and return a cost value.
        - an Optimisation object which provides a specific optimisation trace overlaid on the cost landscape.
    gradient : bool, optional
        If True, the gradient is shown (default: False).
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

    # Assign input as a cost or optimisation object
    if isinstance(cost_or_optim, (pybop.BaseOptimiser, pybop.Optimisation)):
        optim = cost_or_optim
        plot_optim = True
        cost = optim.cost
    else:
        cost = cost_or_optim
        plot_optim = False

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

    if gradient:
        grad_parameter_costs = []

        # Determine the number of gradient outputs from cost.evaluateS1
        num_gradients = len(cost.evaluateS1(np.array([x[0], y[0]]))[1])

        # Create an array to hold each gradient output & populate
        grads = [np.zeros((len(y), len(x))) for _ in range(num_gradients)]
        for i, xi in enumerate(x):
            for j, yj in enumerate(y):
                (*current_grads,) = cost.evaluateS1(np.array([xi, yj]))[1]
                for k, grad_output in enumerate(current_grads):
                    grads[k][j, i] = grad_output

        # Append the arrays to the grad_parameter_costs list
        grad_parameter_costs.extend(grads)

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

    if plot_optim:
        # Plot the optimisation trace
        optim_trace = np.array([item for sublist in optim.log for item in sublist])
        optim_trace = optim_trace.reshape(-1, 2)
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

        # Plot the initial guess
        if optim.x0 is not None:
            fig.add_trace(
                go.Scatter(
                    x=[optim.x0[0]],
                    y=[optim.x0[1]],
                    mode="markers",
                    marker_symbol="circle",
                    marker=dict(
                        color="mediumspringgreen",
                        line_color="mediumspringgreen",
                        line_width=1,
                        size=14,
                        showscale=False,
                    ),
                    showlegend=False,
                )
            )

    # Update the layout and display the figure
    fig.update_layout(**layout_kwargs)
    if "ipykernel" in sys.modules and show:
        fig.show("svg")
    elif show:
        fig.show()

    if gradient:
        grad_figs = []
        for i, grad_costs in enumerate(grad_parameter_costs):
            # Update title for gradient plots
            updated_layout_options = layout_options.copy()
            updated_layout_options["title"] = f"Gradient for Parameter: {i+1}"

            # Create contour plot with updated layout options
            grad_layout = go.Layout(updated_layout_options)

            # Create fig
            grad_fig = go.Figure(
                data=[go.Contour(x=x, y=y, z=grad_costs)], layout=grad_layout
            )
            grad_fig.update_layout(**layout_kwargs)

            if "ipykernel" in sys.modules and show:
                grad_fig.show("svg")
            elif show:
                grad_fig.show()

            # append grad_fig to list
            grad_figs.append(grad_fig)

        return fig, grad_figs

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
    bounds = np.empty((len(cost.parameters), 2))
    for i, param in enumerate(cost.parameters):
        if param.bounds is not None:
            bounds[i] = param.bounds
        else:
            raise ValueError("plot2d could not find bounds required for plotting")

    return bounds
