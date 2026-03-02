from typing import TYPE_CHECKING

from pybop.plot import PlotlyManager

if TYPE_CHECKING:
    from pybop.samplers.base_pints_sampler import SamplingResult


def trace(result: "SamplingResult", **kwargs):
    """
    Plot trace plots for the posterior samples.
    """
    # Import plotly only when needed
    go = PlotlyManager().go

    for i in range(result.n_parameters):
        fig = go.Figure()

        for j, chain in enumerate(result.chains):
            fig.add_trace(go.Scatter(y=chain[:, i], mode="lines", name=f"Chain {j}"))

        fig.update_layout(
            title=f"Parameter {i} Trace Plot",
            xaxis_title="Sample Index",
            yaxis_title="Value",
        )
        fig.update_layout(**kwargs)
        fig.show()


def chains(result: "SamplingResult", **kwargs):
    """
    Plot posterior distributions for each chain.
    """
    # Import plotly only when needed
    go = PlotlyManager().go

    fig = go.Figure()

    for i, chain in enumerate(result.chains):
        for j in range(chain.shape[1]):
            fig.add_trace(
                go.Histogram(
                    x=chain[:, j],
                    name=f"Chain {i} - Parameter {j}",
                    opacity=0.75,
                )
            )

            fig.add_shape(
                type="line",
                x0=result.mean[j],
                y0=0,
                x1=result.mean[j],
                y1=result.max[j],
                name=f"Mean - Parameter {j}",
                line=dict(color="Black", width=1.5, dash="dash"),
            )

    fig.update_layout(
        barmode="overlay",
        title="Posterior Distribution",
        xaxis_title="Value",
        yaxis_title="Density",
    )
    fig.update_layout(**kwargs)
    fig.show()


def posterior(result: "SamplingResult", **kwargs):
    """
    Plot the summed posterior distribution across chains.
    """
    # Import plotly only when needed
    go = PlotlyManager().go

    fig = go.Figure()

    for j in range(result.all_samples.shape[1]):
        histogram = go.Histogram(
            x=result.all_samples[:, j],
            name=f"Parameter {j}",
            opacity=0.75,
        )
        fig.add_trace(histogram)
        fig.add_vline(
            x=result.mean[j], line_width=3, line_dash="dash", line_color="black"
        )

    fig.update_layout(
        barmode="overlay",
        title="Posterior Distribution",
        xaxis_title="Value",
        yaxis_title="Density",
    )
    fig.update_layout(**kwargs)
    fig.show()
    return fig


def summary_table(result: "SamplingResult"):
    """
    Display summary statistics in a table.
    """
    # Import plotly only when needed
    go = PlotlyManager().go

    summary_stats = result.get_summary_statistics()

    header = ["Statistic", "Value"]
    values = [
        ["Mean", summary_stats["mean"]],
        ["Median", summary_stats["median"]],
        ["Standard Deviation", summary_stats["std"]],
        ["95% CI Lower", summary_stats["ci_lower"]],
        ["95% CI Upper", summary_stats["ci_upper"]],
    ]

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(values=header),
                cells=dict(
                    values=[[row[0] for row in values], [row[1] for row in values]]
                ),
            )
        ]
    )

    fig.update_layout(title="Summary Statistics")
    fig.show()
