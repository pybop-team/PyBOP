from pybop import StandardPlot


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

    # Extract log from the optimisation object
    cost_log = optim.log["cost"]

    # Generate a list of iteration numbers
    iteration_numbers = list(range(1, len(cost_log) + 1))

    # Create a plotting dictionary
    plot_dict = StandardPlot(
        x=iteration_numbers,
        y=cost_log,
        layout_options=dict(
            xaxis_title="Iteration",
            yaxis_title="Cost",
            title="Convergence",
        ),
        trace_names=optim.name(),
    )

    # Generate and display the figure
    fig = plot_dict(show=False)
    fig.update_layout(**layout_kwargs)
    if show:
        fig.show()

    return fig
