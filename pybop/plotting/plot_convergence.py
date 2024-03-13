import sys
import pybop
import numpy as np


def plot_convergence(optim, show=True, **layout_kwargs):
    """
    Plot the convergence of the optimisation algorithm.

    Parameters
    -----------
    optim : object
        Optimisation object containing the cost function and optimiser.
    show : bool, optional
        If True, the figure is shown upon creation (default: True).
    **layout_kwargs : optional
        Valid Plotly layout keys and their values,
        e.g. `xaxis_title="Time [s]"` or
        `xaxis={"title": "Time [s]", "titlefont_size": 18}`.

    Returns
    ---------
    fig : plotly.graph_objs.Figure
        The Plotly figure object for the convergence plot.
    """

    # Extract the cost function and log from the optimisation object
    cost = optim.cost
    log = optim.log

    # Compute the minimum cost for each iteration
    min_cost_per_iteration = [
        min(cost(solution) for solution in log_entry) for log_entry in log
    ]

    # Generate a list of iteration numbers
    iteration_numbers = list(range(1, len(min_cost_per_iteration) + 1))

    # Create a plotting dictionary
    plot_dict = pybop.StandardPlot(
        x=iteration_numbers,
        y=min_cost_per_iteration,
        layout_options=dict(
            xaxis_title="Iteration", yaxis_title="Cost", title="Convergence"
        ),
        trace_names=optim.optimiser.name(),
    )

    # Generate and display the figure
    fig = plot_dict(show=False)
    fig.update_layout(**layout_kwargs)
    if "ipykernel" in sys.modules and show:
        fig.show("svg")
    elif show:
        fig.show()

    return fig


def plot_optim2d(optim, bounds=None, steps=10, show=True, **layout_kwargs):
    """
    Plot a 2D visualization of a cost landscape using Plotly with the optimisation trace.

    This function generates a contour plot representing the cost landscape for a provided
    callable cost function over a grid of parameter values within the specified bounds.

    Parameters
    ----------
    optim : object
        Optimisation object which provides a specific optimisation trace overlaid on the cost landscape.
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
    # Extract the cost function from the optimisation object
    cost = optim.cost

    # Create the cost landscape
    fig = pybop.plot_cost2d(cost, bounds=bounds, steps=steps, show=False)

    # Import plotly only when needed
    go = pybop.PlotlyManager().go

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

    return fig
