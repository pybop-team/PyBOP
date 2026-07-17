from pybop.plot.util import get_backend_from_figure, parse_data, wrap_text


def trajectories(
    x,
    y,
    title: str = None,
    xaxis_title: str = None,
    yaxis_title: str = None,
    labels=None,
    label_width=20,
    show=True,
    backend=None,
    figures=None,
    axes=None,
):
    """
    Quickly plot one or more trajectories.

    Parameters
    ----------
    x : list or np.ndarray
        X-axis data points.
    y : list or np.ndarray
        Y-axis data points for each trajectory.
    title: str, optional
        The title of the figure
    xaxis_title: str, optional
        Sets the title/label of the x-axis
    yaxis_title: str, optional
        Sets the title/label of the y-axis
        Settings to modify the default trace type (default: DEFAULT_TRACE_OPTIONS).
    labels : list or str, optional
        Name(s) for the trace(s) (default: None).
    label_width : int, optional
        Maximum length of the labels before text wrapping is used (default: 20).
    show : bool, optional
        If True, the figure is shown upon creation (default: True).
    backend: str or pybop.plot.backends.PlotBackend, optional
        The plotting backend to be used.
    figures: figure object, optional
        Figure for plotting. If not provided a new figure is created
    axes: axis, optional
        Thes axis to be used for plotting
        plotly: axis expected to be of the form tuple(row, col)

    Returns
    -------
    fig : if show is False; plotly.graph_objs.Figure or matplotlib.figure.Figure
    None : if show is True
    """
    backend = get_backend_from_figure(backend, figures)
    figures, axes, create_figure, _ = backend.parse_input_axes(
        figures, axes, num_plots=1
    )

    if create_figure:
        fig = backend.create_figure(
            style={"height": 600, "width": 600, "bg_color": "white"}
        )
    else:
        fig = figures[0]

    backend.update_axes_titles(fig, axes[0], xaxis_title, yaxis_title)
    backend.update_plot_titles(fig, axes[0], title)

    x, y = parse_data(x, y)
    xi = x[0]
    for i in range(0, len(y)):
        if len(x) > 1:
            xi = x[i]
        label = None
        if labels is not None:
            label = wrap_text(labels[i], label_width, backend=backend.name)

        backend.plot_trace(backend.line(xi, y[i], label), fig, ax=axes[0])

    backend.legend(fig, axes=axes[0])

    if show:
        backend.show_figure(fig)
    else:
        return fig
