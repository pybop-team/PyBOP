from typing import TYPE_CHECKING

from pybop.plot.standard_plots import StandardPlot

if TYPE_CHECKING:
    from pybop._result import Result


def convergence(result: "Result", show=True, **layout_kwargs):
    """
    Plot the convergence of the optimisation algorithm.

    Parameters
    -----------
    result : pybop.Result
        Optimisation result containing the history of parameter values and associated cost.
    show : bool, optional
        If True, the figure is shown upon creation (default: True).
    **layout_kwargs : optional
        Valid Plotly layout keys and their values,
        e.g. `xaxis_title="Time [s]"` or
        `xaxis={"title": "Time [s]", font={"size":14}}`

    Returns
    ---------
    fig : plotly.graph_objs.Figure
        The Plotly figure object for the convergence plot.
    """

    # Extract log from the optimisation object
    cost_log = result.cost_convergence

    # Generate a list of iteration numbers
    iteration_numbers = list(range(1, len(cost_log) + 1))

    # Create a plot dictionary
    plot_dict = StandardPlot(
        x=iteration_numbers,
        y=cost_log,
        layout_options=dict(
            xaxis_title="Evaluation",
            yaxis_title="Cost",
            title="Convergence",
        ),
        trace_names=result.method_name,
    )

    # Generate and display the figure
    fig = plot_dict(show=False)
    fig.update_layout(**layout_kwargs)
    if show:
        fig.show()

    return fig
