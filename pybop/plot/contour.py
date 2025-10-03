import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from pybop.plot.plotly_manager import PlotlyManager
from pybop.problems.problem import Problem

if TYPE_CHECKING:
    from pybop._result import OptimisationResult


def contour(
    call_object: "Problem | OptimisationResult",
    gradient: bool = False,
    bounds: np.ndarray | None = None,
    transformed: bool = False,
    steps: int = 10,
    show: bool = True,
    **layout_kwargs,
):
    """
    Plot a 2D visualisation of a cost landscape using Plotly.

    This function generates a contour plot representing the cost landscape for a provided
    callable cost function over a grid of parameter values within the specified bounds.

    Parameters
    ----------
    call_object : pybop.Problem | pybop.OptimisationResult
        Either:
        - the cost function to be evaluated. Must accept a list of parameter values and return a cost value.
        - an optimiser result which provides a specific optimisation trace overlaid on the cost landscape.
    gradient : bool, optional
        If True, the gradient is shown (default: False).
    bounds : numpy.ndarray | list[list[float]], optional
        A 2x2 array specifying the [min, max] bounds for each parameter. If None, uses
        `parameters.get_bounds_for_plotly`.
    transformed : bool, optional
        Uses the transformed parameter values (as seen by the optimiser) for plotting.
    steps : int, optional
        The number of grid points to divide the parameter space into along each dimension (default: 10).
    show : bool, optional
        If True, the figure is shown upon creation (default: True).
    **layout_kwargs : optional
        Valid Plotly layout keys and their values,
        e.g. `xaxis_title="Time [s]"` or
        `xaxis={"title": "Time [s]", font={"size":14}}`

    Returns
    -------
    plotly.graph_objs.Figure
        The Plotly figure object containing the cost landscape plot.

    Raises
    ------
    ValueError
        If the cost function does not return a valid cost when called with a parameter list.
    """
    plot_optim = False
    problem = call_object

    # Assign input as a cost or optimisation result
    if not isinstance(call_object, Callable):
        plot_optim = True
        result = call_object
        problem = result.optim.problem

    parameters = problem.parameters
    names = parameters.names
    additional_values = []

    if len(parameters) < 2:
        raise ValueError("This cost function takes fewer than 2 parameters.")

    if len(parameters) > 2:
        warnings.warn(
            "This cost function requires more than 2 parameters. "
            "Plotting in 2d with fixed values for the additional parameters.",
            UserWarning,
            stacklevel=2,
        )
        for (
            i,
            param,
        ) in enumerate(parameters):
            if i > 1:
                additional_values.append(param.current_value)
                print(f"Fixed {param.name}:", param.current_value)

    # Set up parameter bounds
    if bounds is None:
        bounds = parameters.get_bounds_for_plotly()
    else:
        bounds = np.asarray(bounds)

    # Generate grid
    x = np.linspace(bounds[0, 0], bounds[0, 1], steps)
    y = np.linspace(bounds[1, 0], bounds[1, 1], steps)

    # Initialize cost matrix
    costs = np.zeros((len(y), len(x)))

    if gradient:
        grad_parameter_costs = []

        # Determine the number of gradient outputs from cost.compute
        num_gradients = problem(
            np.asarray([x[0], y[0]] + additional_values),
            calculate_sensitivities=True,
        )[1].shape[0]

        # Create an array to hold each gradient output
        grads = [np.zeros((len(y), len(x))) for _ in range(num_gradients)]

    # Populate cost matrix
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            if gradient:
                costs[j, i], (*current_grads,) = problem(
                    np.asarray([xi, yj] + additional_values),
                    calculate_sensitivities=True,
                )
                for k, grad_output in enumerate(current_grads):
                    grads[k][j, i] = grad_output
            else:
                costs[j, i] = problem(
                    np.asarray([xi, yj] + additional_values),
                )

    # Append the arrays to the grad_parameter_costs list
    if gradient:
        grad_parameter_costs.extend(grads)

    # Apply any transformation if requested
    def transform_array_of_values(list_of_values, parameter):
        """Apply transformation if requested."""
        if transformed:
            return np.asarray(
                [parameter.transformation.to_search(value) for value in list_of_values]
            ).flatten()
        return list_of_values

    x = transform_array_of_values(x, parameters[names[0]])
    y = transform_array_of_values(y, parameters[names[1]])
    bounds[0] = transform_array_of_values(bounds[0], parameters[names[0]])
    bounds[1] = transform_array_of_values(bounds[1], parameters[names[1]])

    # Import plotly only when needed
    go = PlotlyManager().go

    # Set default layout properties
    layout_options = dict(
        title="Cost Landscape",
        title_x=0.5,
        title_y=0.905,
        width=600,
        height=600,
        xaxis=dict(range=bounds[0], showexponent="last", exponentformat="e"),
        yaxis=dict(range=bounds[1], showexponent="last", exponentformat="e"),
        legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=1),
    )
    layout_options["xaxis_title"] = (
        "Transformed " + names[0] if transformed else names[0]
    )
    layout_options["yaxis_title"] = (
        "Transformed " + names[1] if transformed else names[1]
    )
    layout = go.Layout(layout_options)

    # Create contour plot and update the layout
    fig = go.Figure(
        data=[go.Contour(x=x, y=y, z=costs, colorscale="Viridis", connectgaps=True)],
        layout=layout,
    )

    if plot_optim:
        # Plot the optimisation trace
        optim_trace = np.asarray([item[:2] for item in result.x_model])
        optim_trace = optim_trace.reshape(-1, 2)

        fig.add_trace(
            go.Scatter(
                x=transform_array_of_values(optim_trace[:, 0], parameters[names[0]]),
                y=transform_array_of_values(optim_trace[:, 1], parameters[names[1]]),
                mode="markers",
                marker=dict(
                    color=[i / len(optim_trace) for i in range(len(optim_trace))],
                    colorscale="Greys",
                    size=8,
                    showscale=False,
                ),
                showlegend=False,
            )
        )

        # Plot the initial guess
        if len(result.x_model) > 0:
            x0 = result.x_model[0]
            fig.add_trace(
                go.Scatter(
                    x=transform_array_of_values([x0[0]], parameters[names[0]]),
                    y=transform_array_of_values([x0[1]], parameters[names[1]]),
                    mode="markers",
                    marker_symbol="x",
                    marker=dict(
                        color="white",
                        line_color="black",
                        line_width=1,
                        size=14,
                        showscale=False,
                    ),
                    name="Initial values",
                )
            )

        # Plot optimised value
        if result.x is not None:
            x_best = result.x
            fig.add_trace(
                go.Scatter(
                    x=transform_array_of_values([x_best[0]], parameters[names[0]]),
                    y=transform_array_of_values([x_best[1]], parameters[names[1]]),
                    mode="markers",
                    marker_symbol="cross",
                    marker=dict(
                        color="black",
                        line_color="white",
                        line_width=1,
                        size=14,
                        showscale=False,
                    ),
                    name="Final values",
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
            updated_layout_options["title"] = f"Gradient for Parameter: {i + 1}"

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
