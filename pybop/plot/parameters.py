import math
from typing import TYPE_CHECKING

from pybop.plot.util import get_backend_from_figure, parse_data

if TYPE_CHECKING:
    from pybop._result import Result


def parameters(
    result: "Result",
    title: str = "Parameter Convergence",
    show: bool = True,
    backend: str = None,
    figures=None,
    axes=None,
):
    """
    Plot the evolution of parameters during the optimisation process using Plotly.

    Parameters
    ----------
    result : pybop.Result
        Optimisation result containing the history of parameter values and associated cost.
    title : str, optional
        The title of the plot (default: "Parameter Convergence").
    show : bool, optional
        If True, the figure is shown upon creation (default: True).
    backend: str or pybop.plot.backends.PlotBackend, optional
        The plotting backend to be used
    figures: figure object , optional
        Figure for plotting. If not provided a new figure is created.
        Can be a single figure or one figure per parameter.
    axes: single axis or list of axes, optional
        axes for plotting
        plotly: axes expected to be of the form tuple(row, col)
        Number of axis must either agree with the number of parameters or
        be a single axis for all parameters.

    Returns
    -------
    fig : if show is False; plotly.graph_objs.Figure or matplotlib.figure.Figure
        The figure object for the parameter plot.
        Returns a list of figures if multiple figures are provided for plotting.
    None : if show is True
    """
    # import plotting backend
    backend = get_backend_from_figure(backend, figures)

    # Extract parameters and log from the optimisation object
    parameters = result.problem.parameters
    x = list(range(len(result.x_model)))
    y = [list(item) for item in zip(*result.x_model, strict=False)]
    x, y = parse_data(x, y)

    # Create lists of axis titles and trace names
    xaxis_titles = []
    yaxis_titles = []
    labels = parameters.names

    figures, axes, create_figure, _ = backend.parse_input_axes(
        figures, axes, num_plots=len(labels)
    )

    for name in labels:
        xaxis_titles.append("Evaluation")
        yaxis_titles.append(
            name if axes[0] is None or len(axes) == len(labels) else "Parameter Value"
        )

    # legend style
    style = (
        {
            "fig_legend": True,
            "outside": ("right", 0.18),
        }
        if figures is None
        else {}
    )

    if create_figure:
        # Create a subplot for each parameter
        num_cols = int(math.ceil(math.sqrt(len(labels))))
        num_rows = int(math.ceil(len(labels) / num_cols))
        fig, axes = backend.make_subplots(
            num_rows=num_rows,
            num_cols=num_cols,
            num_plots=len(labels),
            title=title,
            style=dict(bg_color="white", width=1600, height=800),
        )
        figures = [fig]

    backend.update_axes_titles(figures, axes, xaxis_titles, yaxis_titles)
    for i in range(len(labels)):
        backend.plot_trace(
            backend.line(x[i % len(x)], y[i], labels[i]),
            figures[i % len(figures)],
            ax=axes[i % len(axes)],
        )

    # add legend
    for i, ax in enumerate(axes):
        backend.legend(figures[i % len(figures)], style=style, axes=ax)

    # Generate the figure and update the layout
    if show:
        backend.show_figure(figures)
    else:
        return figures[0] if len(figures) == 1 else figures
