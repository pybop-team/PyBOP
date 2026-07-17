from typing import TYPE_CHECKING

import numpy as np

from pybop.plot.util import get_backend_from_figure, remove_brackets
from pybop.problems.meta_problem import MetaProblem
from pybop.simulators.failed_solution import FailedSolution

if TYPE_CHECKING:
    from pybop.optimisers.ep_bolfi_optimiser import BayesianOptimisationResult
    from pybop.samplers.base_sampler import SamplingResult


def predictive(
    result: "BayesianOptimisationResult | SamplingResult",
    number_of_traces: int = 8,
    data_legend_entry=None,
    rvs_legend_entry=None,
    pdf_plot=None,
    pdf_label: str = "PDF",
    colour_scale="viridis",
    show: bool = True,
    backend: str | None = None,
    figures=None,
    axes=None,
):
    """
    Plot the predictive posterior of a Bayesian optimisation result.

    Parameters
    ----------
    result : pybop.BayesianOptimisationResult or pybop.SamplingResult
        The result of the Bayesian optimisation or sampling process.
    number_of_traces : int, optional
        The number of posterior predictive traces to plot (default: 8).
    data_legend_entry : str, optional
        The legend entry for the observed data (default: None).
    rvs_legend_entry : str, optional
        The legend entry for the random variable samples (default: None).
    pdf_plot : tuple, optional
        A tuple containing the x and y values for a PDF plot to overlay on the predictive plot (default: None).
    pdf_label : str, optional
        The label for the PDF plot (default: "PDF").
    colour_scale : str, optional
        The colour scale to use for the predictive traces (default: "viridis").
    show : bool, optional
        If True, the figure is shown upon creation (default: True).
    backend : str or pybop.plot.backends.PlotBackend, optional
        Select a plotting backend. If None, the current default backend is used.
    figures: figure object, optional
        Figure for plotting. If not provided a new figure is created for each problem.
        Can be a single figure or one figure per problem.
    axes: axis, optional
        The axes to be used for plotting. One axis per problem is expected.
        plotly: axes expected to be of the form list of tuple(row, col)

    Returns
    -------
    None: if show is True
    Figure or list of figures: if show is False
        If show is False, returns a single figure or a list of figures containing the predictive posterior
        for each problem.
    """

    # Create a plot for each problem
    problems = (
        result.problem.problems
        if isinstance(result.problem, MetaProblem)
        else [result.problem]
    )

    # Import plotting backend
    backend = get_backend_from_figure(backend, figures)

    # Process input figures
    figures, axes, create_figure, _ = backend.parse_input_axes(
        figures, axes, num_plots=len(problems), allow_single_axis=False
    )

    # Retrieve data for plotting
    posterior_samples = result.posterior.sample_from_distribution(
        n_samples=number_of_traces
    )
    posterior_samples_pdf = np.asarray(
        [result.posterior.distribution.pdf(s) for s in posterior_samples]
    )
    pdf_range = np.asarray([posterior_samples_pdf.min(), posterior_samples_pdf.max()])

    for i, problem in enumerate(problems):
        if create_figure:
            fig = backend.create_figure(
                style={"bg_color": "white", "width": 600, "height": 600},
            )
            figures = np.append(figures, fig)
            ax = None
        else:
            fig = figures[i]
            ax = axes[i]

        backend.update_axes_titles(
            fig, ax, remove_brackets(problem.domain), remove_brackets(problem.target[0])
        )
        backend.plot_trace(
            backend.line(
                x=problem.domain_data,
                y=problem.target_data[problem.target[0]],
                label=data_legend_entry,
            ),
            fig,
            ax=ax,
        )

        # Simulate the samples and add to plot
        inputs = [problem.parameters.to_dict(s) for s in posterior_samples]
        simulations = problem.simulate_batch(inputs=inputs)
        for pdf, sim in zip(posterior_samples_pdf, simulations, strict=False):
            if not isinstance(sim, FailedSolution):
                colors = backend.sample_color_scale(
                    pdf, d_min=pdf_range[0], d_max=pdf_range[1]
                )
                backend.plot_trace(
                    backend.line(
                        x=problem.domain_data,
                        y=sim[problem.target[0]].data,
                        style=dict(color=colors[0], linestyle="dotted"),
                    ),
                    fig,
                    ax=ax,
                )

        # Add the colourbar
        backend.colorbar(
            fig, pdf_range, colorscale=colour_scale, label="Posterior PDF", ax=ax
        )

        if pdf_plot is not None:
            backend.plot_trace(
                backend.line(
                    x=pdf_plot[0],
                    y=pdf_plot[1],
                    label=pdf_label,
                ),
                fig,
                ax=ax,
            )
        if show:
            backend.show_figure(fig)

    if not show:
        return figures[0] if len(figures) == 1 else figures
