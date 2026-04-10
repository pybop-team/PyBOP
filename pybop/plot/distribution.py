import numpy as np

from pybop.parameters.parameter import Parameters
from pybop.plot.standard_plots import StandardSubplot


def distribution(
    parameters: Parameters,
    posterior: Parameters | None = None,
    n_samples: int = 100,
    transformed: bool = False,
    show: bool = True,
    **layout_kwargs,
):
    """
    Plot the posterior on top of the prior distribution for a Bayesian optimisation result.
    """
    # Create lists of axis titles and trace names
    axis_titles = []
    trace_names = (
        parameters.names
        if posterior is None
        else ["Prior"] * len(parameters) + ["Posterior"] * len(parameters)
    )
    for name in parameters.names:
        axis_titles.append(
            (name + " (transformed)" if transformed else name, "Probability density")
        )

    # Evaluate marginal distributions for each parameter
    values = []
    probability = []
    for p in parameters:
        d = p.transformed_distribution if transformed else p.distribution
        samples = d.rvs(size=n_samples)
        parameter_range = np.linspace(min(samples), max(samples), n_samples)
        values.append(parameter_range)
        probability.append([d.pdf(s) for s in values[-1]])

    # Set subplot layout options
    layout_options = dict(
        width=1024,
        height=576,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # Create a plot dictionary
    plot_dict = StandardSubplot(
        x=values,
        y=probability,
        axis_titles=axis_titles,
        layout_options=layout_options,
        trace_names=trace_names,
        trace_name_width=50,
    )
    fig = plot_dict(show=False)

    if posterior is not None:
        for idx, p in enumerate(posterior):
            d = p.transformed_distribution if transformed else p.distribution
            samples = d.rvs(size=n_samples)
            parameter_range = np.linspace(min(samples), max(samples), n_samples)
            values.append(parameter_range)
            probability.append([d.pdf(s) for s in values[-1]])

            trace = plot_dict.create_trace(
                values[-1], probability[-1], **plot_dict.trace_options
            )
            row = (idx // plot_dict.num_cols) + 1
            col = (idx % plot_dict.num_cols) + 1
            fig.add_trace(trace, row=row, col=col)

    fig.update_layout(**layout_kwargs)
    if show:
        fig.show()

    return fig
