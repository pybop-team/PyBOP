import sys

import numpy as np

import pybop


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

    # Find the best cost from each iteration
    best_cost_per_iteration = [
        min((cost(solution) for solution in log_entry), default=np.inf)
        if optim.minimising
        else max((cost(solution) for solution in log_entry), default=-np.inf)
        for log_entry in log
    ]

    # Generate a list of iteration numbers
    iteration_numbers = list(range(1, len(best_cost_per_iteration) + 1))

    # Create a plotting dictionary
    plot_dict = pybop.StandardPlot(
        x=iteration_numbers,
        y=best_cost_per_iteration,
        layout_options=dict(
            xaxis_title="Iteration", yaxis_title="Cost", title="Convergence"
        ),
        trace_names=optim.name(),
    )

    # Generate and display the figure
    fig = plot_dict(show=False)
    fig.update_layout(**layout_kwargs)
    if "ipykernel" in sys.modules and show:
        fig.show("svg")
    elif show:
        fig.show()

    return fig
