from pybop import StandardPlot, plot_trajectories


def plot_dataset(dataset, signal=None, trace_names=None, show=True, **layout_kwargs):
    """
    Quickly plot a PyBOP Dataset using Plotly.

    Parameters
    ----------
    dataset : object
        A PyBOP dataset.
    signal : list or str, optional
        The name of the time series to plot (default: "Voltage [V]").
    trace_names : list or str, optional
        Name(s) for the trace(s) (default: "Data").
    show : bool, optional
        If True, the figure is shown upon creation (default: True).
    **layout_kwargs : optional
        Valid Plotly layout keys and their values,
        e.g. `xaxis_title="Time / s"` or
        `xaxis={"title": "Time / s", "titlefont_size": 18}`.

    Returns
    -------
    plotly.graph_objs.Figure
        The Plotly figure object for the scatter plot.
    """

    # Get data dictionary
    if signal is None:
        signal = ["Voltage [V]"]
    dataset.check(signal=signal)

    # Compile ydata and labels or legend
    y = [dataset[s] for s in signal]
    if len(signal) == 1:
        yaxis_title = signal[0]
        if trace_names is None:
            trace_names = ["Data"]
    else:
        yaxis_title = "Output"
        if trace_names is None:
            trace_names = StandardPlot.remove_brackets(signal)

    # Create the figure
    fig = plot_trajectories(
        x=dataset["Time [s]"],
        y=y,
        trace_names=trace_names,
        show=False,
        xaxis_title="Time / s",
        yaxis_title=yaxis_title,
    )
    fig.update_layout(**layout_kwargs)
    if show:
        fig.show()

    return fig
