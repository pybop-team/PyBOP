from pybop.plot.trajectories import trajectories
from pybop.plot.util import get_backend_from_figure, remove_brackets


def dataset(
    dataset, signal=None, labels=None, show=True, backend=None, figures=None, axes=None
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

    # Get data dictionary
    if signal is None:
        signal = ["Voltage [V]"]
    dataset.check(signal=signal)

    # Compile ydata and labels or legend
    y = [dataset[s] for s in signal]
    if len(signal) == 1:
        yaxis_title = remove_brackets(signal[0])
        if labels is None:
            labels = ["Data"]
    else:
        yaxis_title = "Output"
        if labels is None:
            labels = remove_brackets(signal)

    # Create the figure
    fig = trajectories(
        x=dataset[dataset.domain],
        y=y,
        labels=labels,
        show=False,
        xaxis_title=remove_brackets(dataset.domain),
        yaxis_title=yaxis_title,
        backend=backend,
        figures=figures,
        axes=axes,
    )

    if show:
        backend.show_figure(fig)
    else:
        return fig
