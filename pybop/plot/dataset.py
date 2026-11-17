from pybop.plot.util import (
    get_backend_from_figure,
    parse_data,
    remove_brackets,
    wrap_text,
)


def dataset(
    dataset,
    signal=None,
    labels=None,
    label_width=20,
    show=True,
    backend=None,
    figures=None,
    axes=None,
):
    """
    Quickly plot a PyBOP Dataset using Plotly.

    Parameters
    ----------
    dataset : object
        A PyBOP dataset.
    signal : list or str, optional
        The name of the time series to plot (default: "Voltage [V]").
    labels : list or str, optional
        Name(s) for the trace(s) (default: "Data").
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
        The figure object for the scatter plot.
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

    # Get data dictionary
    if signal is None:
        signal = ["Voltage [V]"]
    signal = [signal] if isinstance(signal, str) else signal
    dataset.check(signal=signal)

    # Compile ydata and labels or legend
    x = dataset[dataset.domain]
    y = [dataset[s] for s in signal]
    xaxis_title = remove_brackets(dataset.domain)
    if len(signal) == 1:
        yaxis_title = remove_brackets(signal[0])
        if labels is None:
            labels = ["Data"]
    else:
        yaxis_title = "Output"
        if labels is None:
            labels = remove_brackets(signal)
    backend.update_axes_titles(fig, axes[0], xaxis_title, yaxis_title)

    trace_style_model = dict(linewidth=1, marker="o", markersize=6)

    x, y = parse_data(x, y)
    xi = x[0]
    for i in range(0, len(y)):
        if len(x) > 1:
            xi = x[i]
        label = None
        if labels is not None:
            label = wrap_text(labels[i], label_width, backend=backend.name)

        backend.plot_trace(
            backend.line(xi, y[i], label, trace_style_model), fig, ax=axes[0]
        )

    backend.legend(fig, axes=axes[0])

    if show:
        backend.show_figure(fig)
    else:
        return fig
