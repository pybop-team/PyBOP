import numpy as np

from pybop.parameters.parameter import Parameters
from pybop.plot.util import get_backend_from_figure, parse_data


def distribution(
    parameters: Parameters,
    posterior: Parameters | None = None,
    title: str = "Prior and Posterior Distributions",
    n_samples: int = 100,
    transformed: bool = False,
    show: bool = True,
    backend: str = None,
    figures=None,
    axes=None,
):
    """
    Plot the posterior on top of the prior distribution for a Bayesian optimisation result.

    Parameters
    ----------
    parameters : pybop.Parameters
        The prior distribution of the parameters.
    posterior : pybop.Parameters, optional
        The posterior distribution of the parameters (default: None).
    title : str, optional
        The title of the figure (default: "Prior and Posterior Distributions").
    n_samples : int, optional
        The number of samples to use for plotting the distributions (default: 100).
    transformed : bool, optional
        If True, the transformed distributions are plotted (default: False).
    show : bool, optional
        If True, the figure is shown upon creation (default: True).
    backend : str or pybop.plot.backends.PlotBackend, optional
        Select a plotting backend. If None, the current default backend is used.
    figures: figure object, optional
        Figure for plotting. If not provided a new figure is created.
        Can be a single figure or one figure per parameter.
    axes: axis, optional
        The axes to be used for plotting. One axis per parameter is expected.
        plotly: axes expected to be of the form list of tuple(row, col)

    Returns
    -------
    fig : if show is False; plotly.graph_objs.Figure or matplotlib.figure.Figure
        The figure object for the distribution plot.
        Returns a list of figures if multiple figures are provided for plotting.
    None : if show is True
    """

    # Create lists of axis titles and trace names
    xaxis_titles = []
    yaxis_titles = []
    labels = parameters.names if posterior is None else ["Prior"] * len(parameters)
    for name in parameters.names:
        xaxis_titles.append(name + " (transformed)" if transformed else name)
        yaxis_titles.append("Probability density")

    # Evaluate marginal distributions for each parameter
    values = []
    probability = []
    for p in parameters:
        d = p.transformed_distribution if transformed else p.distribution
        samples = d.rvs(size=n_samples)
        parameter_range = np.linspace(min(samples), max(samples), n_samples)
        values.append(parameter_range)
        probability.append([d.pdf(s) for s in values[-1]])

    values, probability = parse_data(values, probability)

    # Get plotting backend
    backend = get_backend_from_figure(backend, figures)

    # Parse figures
    num_plots = len(probability)
    figures, axes, create_figure, _ = backend.parse_input_axes(
        figures, axes, num_plots=len(labels)
    )

    # Create subplots for each parameter
    if create_figure:
        num_cols = int(np.ceil(np.sqrt(num_plots)))
        num_rows = int(np.ceil(num_plots / num_cols))
        fig, axes = backend.make_subplots(
            num_rows=num_rows,
            num_cols=num_cols,
            num_plots=num_plots,
            title=title,
            style={"bg_color": "white", "width": 1600, "height": 800},
        )
        figures = [fig]

    backend.update_axes_titles(figures, axes, xaxis_titles, yaxis_titles)

    for i in range(num_plots):
        backend.plot_trace(
            backend.line(values[i], probability[i], labels[i]),
            figures[i % len(figures)],
            ax=axes[i % len(axes)],
        )

    if posterior is not None:
        for idx, p in enumerate(posterior):
            d = p.transformed_distribution if transformed else p.distribution
            samples = d.rvs(size=n_samples)
            parameter_range = np.linspace(min(samples), max(samples), n_samples)
            values.append(parameter_range)
            probability.append([d.pdf(s) for s in values[-1]])

            line = backend.line(values[-1], probability[-1], label="Posterior")
            ax = axes[idx % len(axes)]
            backend.plot_trace(line, figures[idx % len(figures)], ax=ax)

    for i, ax in enumerate(axes):
        backend.legend(
            figures[i % len(figures)],
            style=dict(
                horizontal=True,
                outside=("top", 0.1),
                loc="lower right",
                coords=(1, 1.02),
                fig_legend=True,
            ),
            axes=ax,
        )
    if show:
        backend.show_figure(figures)
    else:
        return figures[0] if len(figures) == 1 else figures
