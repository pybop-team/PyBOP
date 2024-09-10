import warnings
from typing import Union

import numpy as np
from scipy.interpolate import griddata

from pybop import BaseCost, BaseOptimiser, Optimisation, PlotlyManager


def plot2d(
    cost_or_optim,
    gradient: bool = False,
    bounds: Union[np.ndarray, None] = None,
    steps: int = 10,
    show: bool = True,
    use_optim_log: bool = False,
    **layout_kwargs,
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
        A 2x2 array specifying the [min, max] bounds for each parameter. If None, uses
        `cost.parameters.get_bounds_for_plotly`.
    steps : int, optional
        The number of grid points to divide the parameter space into along each dimension (default: 10).
    show : bool, optional
        If True, the figure is shown upon creation (default: True).
    use_optim_log : bool, optional
        If True, the optimisation log is used to shape the cost landscape (default: False).
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
    if isinstance(cost_or_optim, (BaseOptimiser, Optimisation)):
        optim = cost_or_optim
        plot_optim = True
        cost = optim.cost
    else:
        cost = cost_or_optim
        plot_optim = False

    if isinstance(cost, BaseCost) and len(cost.parameters) < 2:
        raise ValueError("This cost function takes fewer than 2 parameters.")

    additional_values = []
    if isinstance(cost, BaseCost) and len(cost.parameters) > 2:
        warnings.warn(
            "This cost function requires more than 2 parameters. "
            "Plotting in 2d with fixed values for the additional parameters.",
            UserWarning,
            stacklevel=2,
        )
        for (
            i,
            param,
        ) in enumerate(cost.parameters):
            if i > 1:
                additional_values.append(param.value)
                print(f"Fixed {param.name}:", param.value)

    # Set up parameter bounds
    if bounds is None:
        bounds = cost.parameters.get_bounds_for_plotly()

    # Generate grid
    x = np.linspace(bounds[0, 0], bounds[0, 1], steps)
    y = np.linspace(bounds[1, 0], bounds[1, 1], steps)

    # Initialize cost matrix
    costs = np.zeros((len(y), len(x)))

    # Populate cost matrix
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            costs[j, i] = cost(np.asarray([xi, yj] + additional_values))

    if gradient:
        grad_parameter_costs = []

        # Determine the number of gradient outputs from cost.compute
        num_gradients = len(
            cost(np.asarray([x[0], y[0]] + additional_values), calculate_grad=True)[1]
        )

        # Create an array to hold each gradient output & populate
        grads = [np.zeros((len(y), len(x))) for _ in range(num_gradients)]
        for i, xi in enumerate(x):
            for j, yj in enumerate(y):
                (*current_grads,) = cost(
                    np.asarray([xi, yj] + additional_values), calculate_grad=True
                )[1]
                for k, grad_output in enumerate(current_grads):
                    grads[k][j, i] = grad_output

        # Append the arrays to the grad_parameter_costs list
        grad_parameter_costs.extend(grads)

    elif plot_optim and use_optim_log:
        # Flatten the cost matrix and parameter values
        flat_x = np.tile(x, len(y))
        flat_y = np.repeat(y, len(x))
        flat_costs = costs.flatten()

        # Append the optimisation trace to the data
        parameter_log = np.asarray(optim.log["x_best"])
        flat_x = np.concatenate((flat_x, parameter_log[:, 0]))
        flat_y = np.concatenate((flat_y, parameter_log[:, 1]))
        flat_costs = np.concatenate((flat_costs, optim.log["cost"]))

        # Order the parameter values and estimate the cost using interpolation
        x = np.unique(flat_x)
        y = np.unique(flat_y)
        xf, yf = np.meshgrid(x, y)
        costs = griddata((flat_x, flat_y), flat_costs, (xf, yf), method="linear")

    # Import plotly only when needed
    go = PlotlyManager().go

    # Set default layout properties
    layout_options = dict(
        title="Cost Landscape",
        title_x=0.5,
        title_y=0.9,
        width=600,
        height=600,
        xaxis=dict(range=bounds[0], showexponent="last", exponentformat="e"),
        yaxis=dict(range=bounds[1], showexponent="last", exponentformat="e"),
    )
    if hasattr(cost, "parameters"):
        name = cost.parameters.keys()
        layout_options["xaxis_title"] = name[0]
        layout_options["yaxis_title"] = name[1]
    layout = go.Layout(layout_options)

    # Create contour plot and update the layout
    fig = go.Figure(
        data=[go.Contour(x=x, y=y, z=costs, connectgaps=True)], layout=layout
    )

    if plot_optim:
        # Plot the optimisation trace
        optim_trace = np.asarray(
            [item[:2] for sublist in optim.log["x"] for item in sublist]
        )
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
    if show:
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

            if show:
                grad_fig.show()

            # append grad_fig to list
            grad_figs.append(grad_fig)

        return fig, grad_figs

    return fig
