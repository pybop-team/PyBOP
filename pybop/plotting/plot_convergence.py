import pybop


def plot_convergence(
    optim, xaxis_title="Iteration", yaxis_title="Cost", title="Convergence"
):
    """
    Plot the convergence of the optimisation algorithm.

    Parameters
    -----------
    optim : optimisation object
        Optimisation object containing the cost function and optimiser.
    xaxis_title : str, optional
        Title for the x-axis (default is "Iteration").
    yaxis_title : str, optional
        Title for the y-axis (default is "Cost").
    title : str, optional
        Title of the plot (default is "Convergence").

    Returns
    ---------
    fig : plotly.graph_objs.Figure
        The Plotly figure object for the convergence plot.
    """

    # Extract the cost function from the optimisation object
    cost_function = optim.cost

    # Compute the minimum cost for each iteration
    min_cost_per_iteration = [
        min(cost_function(solution) for solution in log_entry)
        for log_entry in optim.log
    ]

    # Generate a list of iteration numbers
    iteration_numbers = list(range(1, len(min_cost_per_iteration) + 1))

    # Create the convergence plot using the StandardPlot class
    fig = pybop.StandardPlot(
        x=iteration_numbers,
        y=min_cost_per_iteration,
        cost=cost_function,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        title=title,
        trace_name=optim.optimiser.name(),
    )()

    # Display the figure
    fig.show()

    return fig
