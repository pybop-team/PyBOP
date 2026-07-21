from typing import TYPE_CHECKING

import numpy as np

from pybop.plot.util import get_backend_from_figure

if TYPE_CHECKING:
    from pybop.samplers.base_pints_sampler import SamplingResult


def chains(result: "SamplingResult", show=True, backend=None, figures=None, axes=None):
    """
    Plot posterior distributions for each chain.

    Parameters
    ----------
    result : pybop.SamplingResult
        The result of the sampling process.
    show : bool, optional
        If True, the figure is shown upon creation (default: True).
    backend : str or pybop.plot.backends.PlotBackend, optional
        Select a plotting backend. If None, the current default backend is used.
    figures: figure object, optional
        Figure for plotting. If not provided a new figure is created.
    axes: axis, optional
        The axes to be used for plotting. A single axis is expected.

    Returns
    -------
    None: if show is True
    Figure object: if show is False
        The figure object for the chain plots.
    """
    # Import backend
    backend = get_backend_from_figure(backend, figures)
    figures, axes, create_figure, _ = backend.parse_input_axes(
        figures, axes, num_plots=1
    )
    fig = backend.create_figure() if create_figure else figures[0]
    ax = axes[0]

    backend.update_axes_titles(fig, ax, "Value", "Density")
    backend.update_plot_titles(fig, ax, "Posterior Distribution")
    parameter_names = result.problem.parameters.names
    for i, chain in enumerate(result.chains):
        for j in range(chain.shape[1]):
            backend.plot_trace(
                backend.histogram_plot(
                    x=chain[:, j],
                    name=f"Chain {i} - {parameter_names[j]}",
                    style=dict(alpha=0.75),
                ),
                fig,
                ax=ax,
            )

            backend.vline(
                fig,
                result.mean[j],
                style=dict(linewidth=1, linestyle="dashed", color="black"),
                ax=ax,
            )

    backend.legend(fig)

    if show:
        backend.show_figure(fig)
    else:
        return fig


def trace(result: "SamplingResult", show=True, backend=None, figures=None, axes=None):
    """
    Plot trace plots for the posterior samples.

    Parameters
    ----------
    result : pybop.SamplingResult
        The result of the sampling process.
    show : bool, optional
        If True, the figure is shown upon creation (default: True).
    backend : str or pybop.plot.backends.PlotBackend, optional
        Select a plotting backend. If None, the current default backend is used.
    figures: figure object, optional
        Figure for plotting. If not provided a new figure is created.
        Can be a single figure or one figure per parameter.
    axes: single axis or list of axes, optional
        axes for plotting
        plotly: axes expected to be of the form tuple(row, col)
        Number of axis must either agree with the number of parameters or
        be a single axis for all parameters.

    Returns
    -------
    None: if show is True
    Figure or list of figures: if show is False
        If show is False, returns a single figure or a list of figures containing the trace plots
        for each parameter.
    """
    # Import plotting backend
    backend = get_backend_from_figure(backend, figures)

    # Process input
    figures, axes, create_figure, single_axis = backend.parse_input_axes(
        figures, axes, num_plots=result.n_parameters
    )

    parameter_names = result.problem.parameters.names
    for i in range(result.n_parameters):
        ax = axes[i % len(axes)]
        title = (
            "Parameter Trace Plot"
            if single_axis
            else f"Trace Plot - {parameter_names[i]}"
        )
        if create_figure:
            fig = backend.create_figure()
            figures = np.append(figures, fig)

        fig = figures[i % len(figures)]
        backend.update_axes_titles(fig, ax, "Sample Index", "Value")
        if i == 0 or not single_axis:
            backend.update_plot_titles(fig, ax, title)

        for j, chain in enumerate(result.chains):
            label = f"{parameter_names[i]} - Chain {j}" if single_axis else f"Chain {j}"
            backend.plot_trace(backend.line(y=chain[:, i], label=label), fig, ax=ax)
        backend.legend(fig)

    if show:
        backend.show_figure(figures)
    else:
        return figures[0] if len(figures) == 1 else figures


def posterior(
    result: "SamplingResult", backend=None, show=True, figures=None, axes=None
):
    """
    Plot the summed posterior distribution across chains.

    Parameters
    ----------
    result : pybop.SamplingResult
        The result of the sampling process.
    show : bool, optional
        If True, the figure is shown upon creation (default: True).
    backend : str or pybop.plot.backends.PlotBackend, optional
        Select a plotting backend. If None, the current default backend is used.
    figures: figure object, optional
        Figure for plotting. If not provided a new figure is created.
    axes: axis, optional
        The axes to be used for plotting. A single axis is expected.

    Returns
    -------
    None: if show is True
    Figure object: if show is False
        The figure object for the posterior distribution plot.
    """
    # Import backend
    backend = get_backend_from_figure(backend, figures)

    # Parse input
    figures, axes, create_figure, _ = backend.parse_input_axes(
        figures, axes, num_plots=1
    )
    fig = backend.create_figure() if create_figure else figures[0]
    ax = axes[0]

    backend.update_axes_titles(fig, ax, "Value", "Density")
    backend.update_plot_titles(fig, ax, "Posterior Distribution")
    parameter_names = result.problem.parameters.names

    for j in range(result.all_samples.shape[1]):
        backend.plot_trace(
            backend.histogram_plot(
                x=result.all_samples[:, j],
                name=f"{parameter_names[j]}",
                style=dict(alpha=0.75),
            ),
            fig,
            ax=ax,
        )
        backend.vline(
            fig,
            result.mean[j],
            style=dict(linewidth=1, linestyle="dashed", color="black"),
            ax=ax,
        )

    backend.legend(fig)

    if show:
        backend.show_figure(fig)
    else:
        return fig


def summary_table(
    result: "SamplingResult", backend=None, figures=None, axes=None, show=True
):
    """
    Display summary statistics in a table.

    Parameters
    ----------
    result : pybop.SamplingResult
        The result of the sampling process.
    backend : str or pybop.plot.backends.PlotBackend, optional
        Select a plotting backend. If None, the current default backend is used.
    figures: figure object, optional
        Figure for plotting. If not provided a new figure is created.
    axes: axis, optional
        The axes to be used for plotting. A single axis is expected.
    show : bool, optional
        If True, the figure is shown upon creation (default: True).

    Returns
    -------
    None: if show is True
    Figure object: if show is False
        The figure object for the summary statistics table.
    """

    summary_stats = result.get_summary_statistics()

    header = ["Statistic", "Value"]
    values = [
        ["Mean", summary_stats["mean"]],
        ["Median", summary_stats["median"]],
        ["Standard Deviation", summary_stats["std"]],
        ["95% CI Lower", summary_stats["ci_lower"]],
        ["95% CI Upper", summary_stats["ci_upper"]],
    ]

    backend = get_backend_from_figure(backend, figures)
    figures, axes, create_figure, _ = backend.parse_input_axes(
        figures, axes, num_plots=1
    )
    fig = None if create_figure else figures[0]
    ax = axes[0]
    fig = backend.show_table(
        header=header,
        values=values,
        title="Summary Statistics",
        fig=fig,
        ax=ax,
    )
    if show:
        backend.show_figure(fig)
    else:
        return fig
