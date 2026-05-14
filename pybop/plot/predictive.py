from typing import TYPE_CHECKING

import numpy as np

from pybop.plot.plotly_manager import PlotlyManager
from pybop.plot.standard_plots import StandardPlot
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
    **layout_kwargs,
):
    """
    Plot the predictive posterior of a Bayesian optimisation result.
    """
    # Import plotly only when needed
    px = PlotlyManager().px

    posterior_samples = result.posterior.sample_from_distribution(
        n_samples=number_of_traces
    )
    posterior_samples_pdf = np.asarray(
        [result.posterior.distribution.pdf(s) for s in posterior_samples]
    )
    pdf_range = np.asarray([posterior_samples_pdf.min(), posterior_samples_pdf.max()])

    # Create a plot for each problem
    problems = (
        result.problem.problems
        if isinstance(result.problem, MetaProblem)
        else [result.problem]
    )
    figure_list = []

    for problem in problems:
        plot_dict = StandardPlot(
            x=problem.domain_data,
            y=problem.target_data[problem.target[0]],
            layout_options=dict(
                xaxis_title=StandardPlot.remove_brackets(problem.domain),
                yaxis_title=StandardPlot.remove_brackets(problem.target[0]),
            ),
            trace_names=data_legend_entry,
        )

        # Simulate the samples and add to plot
        inputs = [problem.parameters.to_dict(s) for s in posterior_samples]
        simulations = problem.simulate_batch(inputs=inputs)
        for pdf, sim in zip(posterior_samples_pdf, simulations, strict=False):
            if not isinstance(sim, FailedSolution):
                plot_dict.add_traces(
                    x=problem.domain_data,
                    y=sim[problem.target[0]].data,
                    line={
                        "dash": "dot",
                        "color": px.colors.sample_colorscale(
                            colour_scale,
                            (pdf - pdf_range[0]) / (pdf_range[1] - pdf_range[0]),
                        )[0],
                    },
                )

        # Add the colourbar
        plot_dict.add_traces(
            x=[None],
            y=[None],
            mode="markers",
            marker={
                "size": 0,
                "color": pdf_range,
                "colorscale": colour_scale,
                "showscale": True,
                "colorbar": {"title": {"text": "Posterior PDF", "side": "right"}},
            },
        )

        if pdf_plot is not None:
            plot_dict.add_traces(
                x=pdf_plot[0],
                y=pdf_plot[1],
                trace_names=pdf_label,
            )

        fig = plot_dict(show=False)
        fig.update_layout(**layout_kwargs)
        if show:
            fig.show()

        figure_list.append(fig)

    return figure_list
