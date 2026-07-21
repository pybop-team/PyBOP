import numpy as np

from pybop.parameters.parameter import Inputs
from pybop.plot.util import get_backend_from_figure


def nyquist(
    problem,
    inputs: Inputs = None,
    show=True,
    title="Nyquist Plot",
    backend=None,
    figures=None,
    axes=None,
):
    """
    Generates Nyquist plots for the given problem by evaluating the model's output and target values.

    Parameters
    ----------
    problem : pybop.Problem
        An instance of a problem class that contains the parameters and methods
        for evaluation and target retrieval.
    title: str, optional
        The title of the figure
    inputs : Inputs, optional
        Input parameters for the problem. If not provided, the default parameters from the problem
        instance will be used. These parameters are verified before use (default is None).
    show : bool, optional
        If True, the plots will be displayed.
    backend: str or pybop.plot.backends.PlotBackend, optional
        The plotting backend to be used.
    figures: figure object or list of figure objects, optional
        Either a single figure or the same number of figures as axes.
    axes: single axis or list of axes, optional
        plotly: axes expected to be of the form tuple(row, col)
        Number of axes must agree with number of targets for the problem.

    Returns
    -------
    fig or list of figs : plotly.graph_objs.Figure or matplotlib.figure.Figure
        A single figure or a list of figures containing the Nyquist plots for each target in the
        problem. If show is True, the figures will be displayed and None will be returned.
    Notes
    -----
    - The function extracts the real part of the impedance from the model's output and the real and imaginary parts
      of the impedance from the target output.
    - For each signal in the problem, a Nyquist plot is created with the model's impedance plotted as a scatter plot.
    - An additional trace for the reference (target output) is added to the plot.

    Example
    -------
    >>> problem = pybop.EISProblem()
    >>> nyquist_figures = nyquist(problem, show=True, title="Nyquist Plot", xaxis_title="Real(Z)", yaxis_title="Imag(Z)")
    >>> # The plots will be displayed and nyquist_figures will contain the list of figure objects.
    """
    # Import plotting backend
    backend = get_backend_from_figure(backend, figures)

    # Process input figures
    figures, axes, create_figure, _ = backend.parse_input_axes(
        figures, axes, num_plots=len(problem.target), allow_single_axis=False
    )

    trace_style_model = dict(
        linewidth=2,
        color="#00CC96",
        marker="o",
        markerfacecolor="#00CC96",
    )
    trace_style_reference = dict(
        linestyle="none", marker="o", fillstyle="none", markeredgecolor="#636EFA"
    )

    if not isinstance(inputs, dict):
        inputs = problem.parameters.to_dict(inputs)

    model_output = problem.simulate(inputs)
    domain_data = model_output["Impedance"].data.real
    target_output = problem.target_data

    for i, var in enumerate(problem.target):
        if create_figure:
            fig = backend.create_figure(
                style={"width": 600, "height": 600, "bg_color": "white"},
            )
            figures = np.append(figures, fig)
            ax = None
        else:
            fig = figures[i]
            ax = axes[i]

        backend.update_axes_titles(fig, ax, r"$Z_{re} / \Omega$", r"$-Z_{im} / \Omega$")
        backend.update_plot_titles(fig, ax, title)

        backend.plot_trace(
            backend.line(
                x=domain_data,
                y=-model_output[var].data.imag,
                label="Model",
                style=trace_style_model,
            ),
            fig,
            ax=ax,
        )

        backend.plot_trace(
            backend.line(
                x=target_output[var].real,
                y=-target_output[var].imag,
                label="Reference",
                style=trace_style_reference,
            ),
            fig,
            ax=ax,
        )
        backend.legend(fig, axes=ax)

    if show:
        backend.show_figure(figures)
    else:
        return figures[0] if len(figures) == 1 else figures
