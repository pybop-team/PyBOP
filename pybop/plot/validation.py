from pybop import Dataset, Problem
from pybop.plot import PlotlyManager
from pybop.plot import dataset as plot_dataset


def validation(
    values: list,
    problem: Problem,
    dataset: Dataset,
    signal: str = "Voltage [V]",
    show: bool = True,
    **layout_kwargs,
):
    """
    Plot the model prediction against the target dataset.

    Parameters
    ----------
    values : array-like
        The optimised parameter values.
    problem : pybop.Problem
        The optimisation problem, including a pybamm pipeline.
    dataset : pybop.Dataset
        The target dataset.
    show : bool, optional
        If True, the figure is shown upon creation (default: True).
    **layout_kwargs : optional
        Valid Plotly layout keys and their values,
        e.g. `xaxis_title="Time / s"` or
        `xaxis={"title": "Time [s]", font={"size":14}}`

    Returns
    -------
    plotly.graph_objs.Figure
        The Plotly figure object for the scatter plot.
    """
    # First plot the dataset
    domain = dataset.domain
    fig = plot_dataset(
        dataset, signal=signal, trace_names=["Target"], show=False, **layout_kwargs
    )

    # Run a simulation with the parameter values provided
    problem.pipeline._solver = problem.pipeline.model.default_solver  # noqa: SLF001
    sol = problem.pipeline.solve(problem.parameters.to_dict(values))[0]

    # Add the simulation to the plot
    go = PlotlyManager().go
    fig.add_trace(
        go.Scatter(x=sol[domain].data, y=sol[signal].data, mode="lines", name="Fit")
    )
    if show:
        fig.show()

    return fig
