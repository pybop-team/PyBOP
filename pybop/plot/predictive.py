import numpy as np
import plotly.express as px

from pybop.optimisers.ep_bolfi_optimiser import BayesianOptimisationResult
from pybop.plot.standard_plots import StandardPlot


def predictive(
    result: "BayesianOptimisationResult",
    simulator,
    number_of_traces=8,
    dataset_y="Voltage [V]",
    data_legend_entry=None,
    rvs_legend_entry=None,
    pdf_plot=None,
    pdf_label="PDF",
    colour_scale="viridis",
    show=True,
    **layout_kwargs,
):
    """
    Plot the predictive posterior of a Bayesian parameterisation result.
    """

    posterior_resamples = result.posterior.rvs(number_of_traces, apply_transform=True)
    posterior_resamples_pdf = result.posterior.pdf(posterior_resamples)
    pdf_range = np.asarray(
        [posterior_resamples_pdf.min(), posterior_resamples_pdf.max()]
    )

    simulations = simulator(posterior_resamples)

    plot_dict = StandardPlot(
        x=result.problem.domain_data,
        y=result.problem.cost.dataset[dataset_y],
        layout_options=layout_kwargs,
        trace_names=data_legend_entry,
    )

    for pdf, pred in zip(posterior_resamples_pdf, simulations, strict=False):
        plot_dict.add_traces(
            x=result.problem.domain_data,
            y=pred,
            line={
                "dash": "dot",
                "color": px.colors.sample_colorscale(
                    colour_scale, (pdf - pdf_range[0]) / (pdf_range[1] - pdf_range[0])
                )[0],
            },
        )

    # Add the colourbar.
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
    return fig
