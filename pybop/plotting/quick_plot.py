import numpy as np
import textwrap
import pybop


class StandardPlot:
    """
    A class for creating and displaying a plotly figure that compares a target dataset against a simulated model output.

    This class provides an interface for generating interactive plots using Plotly, with the ability to include an
    optional secondary dataset and visualize uncertainty if provided.

    Attributes:
    -----------
    x : list
        The x-axis data points.
    y : list or np.ndarray
        The primary y-axis data points representing the simulated model output.
    y2 : list or np.ndarray, optional
        An optional secondary y-axis data points representing the target dataset against which the model output is compared.
    cost : float
        The cost associated with the model output.
    title : str, optional
        The title of the plot.
    xaxis_title : str, optional
        The title for the x-axis.
    yaxis_title : str, optional
        The title for the y-axis.
    trace_name : str, optional
        The name of the primary trace representing the model output. Defaults to "Simulated".
    width : int, optional
        The width of the figure in pixels. Defaults to 720.
    height : int, optional
        The height of the figure in pixels. Defaults to 540.

    Example:
    ----------
    >>> x_data = [1, 2, 3, 4]
    >>> y_simulated = [10, 15, 13, 17]
    >>> y_target = [11, 14, 12, 16]
    >>> plot = pybop.StandardPlot(x_data, y_simulated, cost=0.05, y2=y_target,
                            title="Model vs. Target", xaxis_title="X Axis", yaxis_title="Y Axis")
    >>> fig = plot()  # Generate the figure
    >>> fig.show()    # Display the figure in a browser
    """

    def __init__(
        self,
        x,
        y,
        cost,
        y2=None,
        title=None,
        xaxis_title=None,
        yaxis_title=None,
        trace_name=None,
        width=1024,
        height=576,
    ):
        self.x = x if isinstance(x, list) else x.tolist()
        self.y = y
        self.y2 = y2
        self.cost = cost
        self.width = width
        self.height = height
        self.title = title
        self.xaxis_title = xaxis_title
        self.yaxis_title = yaxis_title
        self.trace_name = trace_name or "Simulated"

        if self.y2 is not None:
            self.sigma = np.std(self.y - self.y2)
            self.y_upper = (self.y + self.sigma).tolist()
            self.y_lower = (self.y - self.sigma).tolist()

        # Attempt to import plotly when an instance is created
        self.go = pybop.PlotlyManager().go

    @staticmethod
    def wrap_text(text, width):
        """
        Wrap text to a specified width.

        Parameters:
        -----------
        text: str
            Text to be wrapped.
        width: int
            Width to wrap text to.

        Returns:
        ----------
        str
            Wrapped text with HTML line breaks.
        """
        wrapped_text = textwrap.fill(text, width=width, break_long_words=False)
        return wrapped_text.replace("\n", "<br>")

    def create_layout(self):
        """
        Create the layout for the plot.
        """
        return self.go.Layout(
            title=self.title,
            title_x=0.5,
            xaxis=dict(title=self.xaxis_title, titlefont_size=12, tickfont_size=12),
            yaxis=dict(title=self.yaxis_title, titlefont_size=12, tickfont_size=12),
            legend=dict(x=1, y=1, xanchor="right", yanchor="top", font_size=12),
            showlegend=True,
            autosize=False,
            width=self.width,
            height=self.height,
            margin=dict(l=10, r=10, b=10, t=75, pad=4),
        )

    def create_traces(self):
        """
        Create the traces for the plot.
        """
        traces = []

        wrapped_trace_name = self.wrap_text(self.trace_name, width=40)
        simulated_trace = self.go.Scatter(
            x=self.x,
            y=self.y,
            line=dict(width=4),
            mode="lines",
            name=wrapped_trace_name,
        )

        if self.y2 is not None:
            target_trace = self.go.Scatter(
                x=self.x, y=self.y2, mode="markers", name="Target"
            )
            fill_trace = self.go.Scatter(
                x=self.x + self.x[::-1],
                y=self.y_upper + self.y_lower[::-1],
                fill="toself",
                fillcolor="rgba(255,229,204,0.8)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=False,
            )
            traces.extend([fill_trace, target_trace])

        traces.append(simulated_trace)

        return traces

    def __call__(self):
        """
        Generate the plotly figure.
        """
        layout = self.create_layout()
        traces = self.create_traces()
        fig = self.go.Figure(data=traces, layout=layout)
        return fig


def quick_plot(params, cost, title="Scatter Plot", width=1024, height=576):
    """
    Plot the target dataset against the minimised model output.

    Parameters:
    -----------
    params : array-like
        Optimised parameters.
    cost : cost object
        Cost object containing the problem, dataset, and signal.
    title : str, optional
        Title of the plot (default is "Scatter Plot").
    width : int, optional
        Width of the figure in pixels (default is 720).
    height : int, optional
        Height of the figure in pixels (default is 540).

    Returns:
    ----------
    fig : plotly.graph_objs.Figure
        The Plotly figure object for the scatter plot.
    """

    # Extract the time data and evaluate the model's output and target values
    time_data = cost.problem._dataset["Time [s]"].data
    model_output = cost.problem.evaluate(params)
    target_output = cost.problem.target()

    # Create the figure using the StandardPlot class
    fig = pybop.StandardPlot(
        x=time_data,
        y=model_output,
        cost=cost,
        y2=target_output,
        xaxis_title="Time [s]",
        yaxis_title=cost.problem.signal,
        title=title,
        trace_name="Model",
        width=width,
        height=height,
    )()

    # Display the figure
    fig.show()

    return fig
