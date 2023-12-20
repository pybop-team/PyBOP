import numpy as np
import textwrap
import pybop


class StandardPlot:
    """
    A class for creating and displaying Plotly figures for model output comparison.

    Generates interactive plots comparing simulated model output with an optional target dataset and visualizes uncertainty.

    Parameters
    ----------
    x : list or np.ndarray
        X-axis data points.
    y : list or np.ndarray
        Primary Y-axis data points for simulated model output.
    cost : float
        Cost associated with the model output.
    y2 : list or np.ndarray, optional
        Secondary Y-axis data points for the target dataset (default: None).
    title : str, optional
        Title of the plot (default: None).
    xaxis_title : str, optional
        Title for the x-axis (default: None).
    yaxis_title : str, optional
        Title for the y-axis (default: None).
    trace_name : str, optional
        Name for the primary trace (default: "Simulated").
    width : int, optional
        Width of the figure in pixels (default: 1024).
    height : int, optional
        Height of the figure in pixels (default: 576).
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
        """
        Initialize the StandardPlot object with simulation and optional target data.

        Parameters
        ----------
        x : list or np.ndarray
            X-axis data points.
        y : list or np.ndarray
            Primary Y-axis data points for simulated model output.
        cost : float
            Cost associated with the model output.
        y2 : list or np.ndarray, optional
            Secondary Y-axis data points for target dataset (default: None).
        title : str, optional
            Plot title (default: None).
        xaxis_title : str, optional
            X-axis title (default: None).
        yaxis_title : str, optional
            Y-axis title (default: None).
        trace_name : str, optional
            Name for the primary trace (default: "Simulated").
        width : int, optional
            Figure width in pixels (default: 1024).
        height : int, optional
            Figure height in pixels (default: 576).
        """
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
        Wrap text to a specified width with HTML line breaks.

        Parameters
        ----------
        text : str
            The text to wrap.
        width : int
            The width to wrap the text to.

        Returns
        -------
        str
            The wrapped text.
        """
        wrapped_text = textwrap.fill(text, width=width, break_long_words=False)
        return wrapped_text.replace("\n", "<br>")

    def create_layout(self):
        """
        Create the layout for the Plotly figure.

        Returns
        -------
        plotly.graph_objs.Layout
            The layout for the Plotly figure.
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
        Create traces for the Plotly figure.

        Returns
        -------
        list
            A list of plotly.graph_objs.Scatter objects to be used as traces.
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
        Generate the Plotly figure.

        Returns
        -------
        plotly.graph_objs.Figure
            The generated Plotly figure.
        """
        layout = self.create_layout()
        traces = self.create_traces()
        fig = self.go.Figure(data=traces, layout=layout)
        return fig


def quick_plot(params, cost, title="Scatter Plot", width=1024, height=576):
    """
    Quickly plot the target dataset against minimized model output.

    Parameters
    ----------
    params : array-like
        Optimized parameters.
    cost : object
        Cost object with problem, dataset, and signal attributes.
    title : str, optional
        Title of the plot (default: "Scatter Plot").
    width : int, optional
        Width of the figure in pixels (default: 1024).
    height : int, optional
        Height of the figure in pixels (default: 576).

    Returns
    -------
    plotly.graph_objs.Figure
        The Plotly figure object for the scatter plot.
    """

    # Extract the time data and evaluate the model's output and target values
    time_data = cost.problem._dataset["Time [s]"].data
    model_output = cost.problem.evaluate(params)
    target_output = cost.problem.target()

    for i in range(0, cost.problem.n_outputs):
        # Create the figure using the StandardPlot class
        fig = pybop.StandardPlot(
            x=time_data,
            y=model_output[:, i],
            cost=cost,
            y2=target_output[:, i],
            xaxis_title="Time [s]",
            yaxis_title=cost.problem.signal[i],
            title=title,
            trace_name="Model",
            width=width,
            height=height,
        )()

        # Display the figure
        fig.show()

    return fig
