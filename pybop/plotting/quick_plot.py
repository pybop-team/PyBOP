import numpy as np


def quick_plot(params, cost, width=720, height=540):
    """
    Plot the target dataset against the minimised model output.

    Inputs:
    -------
    x: array
        Optimised parameters
    cost: cost object
        Cost object containing the problem, dataset, and signal
    """

    # Generate the model output
    x = cost.problem._dataset["Time [s]"].data
    y = cost.problem.evaluate(params)
    y2 = cost.problem.target()

    # Create figure
    fig = create_figure(x, y, y2, cost, width, height)

    # Display figure
    fig.show()

    return fig


def create_figure(x, y, y2, cost, width, height):
    # Import plotly only when needed
    import plotly.graph_objs as go

    # Estimate the uncertainty (sigma) of the model output
    x = x.tolist()
    sigma = np.std(y - y2)
    y_upper = (y + sigma).tolist()
    y_lower = (y - sigma).tolist()

    # Create traces for the measured and simulated values
    target_trace = go.Scatter(
        x=x,
        y=y2,
        line=dict(color="rgb(102,102,255,0.1)"),
        mode="markers",
        name="Target",
    )
    simulated_trace = go.Scatter(
        x=x, y=y, line=dict(width=4, color="rgb(255,128,0)"), mode="lines", name="Model"
    )

    # Create a trace for the fill area representing the uncertainty (sigma)
    fill_trace = go.Scatter(
        x=x + x[::-1],
        y=y_upper + y_lower[::-1],
        fill="toself",
        fillcolor="rgba(255,229,204,0.8)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        showlegend=False,
    )

    # Define the layout for the plot
    layout = go.Layout(
        title="Optimised Comparison",
        title_x=0.55,
        title_y=0.9,
        xaxis=dict(title="Time [s]", titlefont_size=12, tickfont_size=12),
        yaxis=dict(title=cost.problem.signal, titlefont_size=12, tickfont_size=12),
        legend=dict(x=0.85, y=1, xanchor="left", yanchor="top", font_size=12),
        showlegend=True,
    )

    # Combine the traces and layout into a figure
    fig = go.Figure(data=[fill_trace, target_trace, simulated_trace], layout=layout)

    # Update the figure to adjust the layout and axis properties
    fig.update_layout(
        autosize=False,
        width=width,
        height=height,
        margin=dict(l=10, r=10, b=10, t=75, pad=4),
    )

    return fig
