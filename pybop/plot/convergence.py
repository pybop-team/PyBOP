from typing import TYPE_CHECKING

from pybop.plot.util import get_backend_from_figure, wrap_text

if TYPE_CHECKING:
    from pybop._result import Result


def convergence(
    result: "Result", show: bool = True, backend: str = None, figures=None, axes=None
):
    """
    Plot the convergence of the optimisation algorithm.

    Parameters
    -----------
    result : pybop.Result
        Optimisation result containing the history of parameter values and associated cost.
    show : bool, optional
        If True, the figure is shown upon creation (default: True).
    backend : str or pybop.plot.backends.PlotBackend, optional
        Select a plotting backend. If None, the current default backend is used.
    figures: figure object, optional
        Figure for plotting. If not provided a new figure is created
    axes: axis, optional
        The axis to be used for plotting
        plotly: axis expected to be of the form tuple(row, col)


    Returns
    ---------
    fig : if show is False; plotly.graph_objs.Figure or matplotlib.figure.Figure
        The figure object for the convergence plot.
    None : if show is True
    """

    # Extract log from the optimisation object
    cost_log = result.cost_convergence

    # Generate a list of iteration numbers
    iteration_numbers = list(range(1, len(cost_log) + 1))

    backend = get_backend_from_figure(backend, figures)

    figures, axes, create_figure, _ = backend.parse_input_axes(
        figures, axes, num_plots=1
    )

    # Create figure
    if create_figure:
        fig = backend.create_figure(
            style={"bg_color": "white", "width": 600, "height": 600},
        )
        ax = None
    else:
        fig = figures[0]
        ax = axes[0]

    backend.update_axes_titles(fig, ax, "Evaluation", "Cost")
    backend.update_plot_titles(fig, ax, "Convergence")

    # Add line plot
    backend.plot_trace(
        backend.line(
            x=iteration_numbers,
            y=cost_log,
            label=wrap_text(result.method_name, width=20, backend=backend.name),
        ),
        fig,
        ax=ax,
    )

    backend.legend(fig, axes=ax)

    # Display or return figure
    if show:
        backend.show_figure(fig)
    else:
        return fig
