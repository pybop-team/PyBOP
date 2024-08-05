import math
import textwrap

import numpy as np

from pybop import PlotlyManager

DEFAULT_LAYOUT_OPTIONS = dict(
    title=None,
    title_x=0.5,
    xaxis=dict(
        title=None,
        showexponent="last",
        exponentformat="e",
        titlefont_size=12,
        tickfont_size=12,
    ),
    yaxis=dict(
        title=None,
        showexponent="last",
        exponentformat="e",
        titlefont_size=12,
        tickfont_size=12,
    ),
    legend=dict(x=1, y=1, xanchor="right", yanchor="top", font_size=12),
    showlegend=True,
    autosize=False,
    width=600,
    height=600,
    margin=dict(l=10, r=10, b=10, t=75, pad=4),
    plot_bgcolor="white",
)
DEFAULT_SUBPLOT_OPTIONS = dict(
    start_cell="bottom-left",
)
DEFAULT_TRACE_OPTIONS = dict(line=dict(width=4), mode="lines")
DEFAULT_SUBPLOT_TRACE_OPTIONS = dict(line=dict(width=2), mode="lines")


class StandardPlot:
    """
    A class for creating and displaying interactive Plotly figures.

    Parameters
    ----------
    x : list or np.ndarray
        X-axis data points.
    y : list or np.ndarray
        Primary Y-axis data points for simulated model output.
    layout : Plotly layout, optional
        A layout for the figure, overrides the layout options (default: None).
    layout_options : dict, optional
        Settings to modify the default layout (default: DEFAULT_LAYOUT_OPTIONS).
    trace_options : dict, optional
        Settings to modify the default trace type (default: DEFAULT_TRACE_OPTIONS).
    trace_names : str, optional
        Name(s) for the primary trace(s) (default: None).
    trace_name_width : int, optional
        Maximum length of the trace names before text wrapping is used (default: 40).

    Returns
    -------
    plotly.graph_objs.Figure
        The generated Plotly figure.
    """

    def __init__(
        self,
        x,
        y,
        layout=None,
        layout_options=None,
        trace_options=None,
        trace_names=None,
        trace_name_width=40,
    ):
        self.x = x
        self.y = y
        self.layout = layout
        self.trace_name_width = trace_name_width

        # Set default layout options and update if provided
        if self.layout is None:
            self.layout_options = DEFAULT_LAYOUT_OPTIONS.copy()
            if layout_options:
                self.layout_options.update(layout_options)

        # Set default trace options and update if provided
        self.trace_options = DEFAULT_TRACE_OPTIONS.copy()
        if trace_options:
            self.trace_options.update(trace_options)

        # Check trace_names and set attribute
        if isinstance(trace_names, str):
            self.trace_names = [trace_names]
        else:
            self.trace_names = trace_names

        # Check type and dimensions of data
        # What we want is a list of 'things plotly can take', e.g. numpy arrays or lists of numbers
        if isinstance(self.x, list):
            # If it's a list of numpy arrays, it's fine
            # If it's a list of lists, it's fine
            # If it's neither, it's a list of numbers that we need to wrap
            if not isinstance(self.x[0], np.ndarray) and not isinstance(
                self.x[0], list
            ):
                self.x = [self.x]
        elif isinstance(self.x, np.ndarray):
            self.x = np.squeeze(self.x)
            if self.x.ndim == 1:
                self.x = [self.x]
            else:
                self.x = self.x.tolist()
        if isinstance(self.y, list):
            if not isinstance(self.y[0], np.ndarray) and not isinstance(
                self.y[0], list
            ):
                self.y = [self.y]
        if isinstance(self.y, np.ndarray):
            self.y = np.squeeze(self.y)
            if self.y.ndim == 1:
                self.y = [self.y]
            else:
                self.y = self.y.tolist()
        if len(self.x) > 1 and len(self.x) != len(self.y):
            raise ValueError(
                "Input x should have either one data series or the same number as y."
            )

        # Attempt to import plotly when an instance is created
        self.go = PlotlyManager().go

        # Create layout
        if self.layout is None:
            self.layout = self.go.Layout(**self.layout_options)

        # Wrap trace names
        if self.trace_names is not None:
            for i, name in enumerate(self.trace_names):
                self.trace_names[i] = self.wrap_text(name, width=self.trace_name_width)

        # Create a trace for each trajectory
        self.traces = []
        x = self.x[0]
        for i in range(0, len(self.y)):
            if len(self.x) > 1:
                x = self.x[i]
            if self.trace_names is not None:
                self.trace_options["name"] = self.trace_names[i]
            else:
                self.trace_options["showlegend"] = False
            trace = self.create_trace(x, self.y[i], **self.trace_options)
            self.traces.append(trace)

    def __call__(self, show=True):
        """
        Generate and show the figure.

        Parameters
        ----------
        show : bool, optional
            If True, the figure is shown upon creation (default: True).
        """
        fig = self.go.Figure(data=self.traces, layout=self.layout)
        if show:
            fig.show()

        return fig

    def create_trace(self, x, y, **trace_options):
        """
        Create a trace for the Plotly figure.

        Returns
        -------
        plotly.graph_objs.Scatter
            A trace for a Plotly figure.
        """

        return self.go.Scatter(
            x=x,
            y=y,
            **trace_options,
        )

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

    @staticmethod
    def remove_brackets(s):
        """
        Remove square brackets from a string and replace with forward slashes
        as per section 7.1 of the SI Handbook
        """
        # If s is an iterable (but not a string), apply the function recursively to each element
        if hasattr(s, "__iter__") and not isinstance(s, str):
            return type(s)(StandardPlot.remove_brackets(i) for i in s)
        elif isinstance(s, str):
            start = s.find("[")
            end = s.find("]")
            if start != -1 and end != -1:
                char_in_brackets = s[start + 1 : end]
                return s[:start] + " / " + char_in_brackets + s[end + 1 :]
        return s


class StandardSubplot(StandardPlot):
    """
    A class for creating and displaying a set of interactive Plotly figures in a grid layout.

    Parameters
    ----------
    x : list or np.ndarray
        X-axis data points.
    y : list or np.ndarray
        Primary Y-axis data points for simulated model output.
    num_rows : int, optional
        Number of rows of subplots, can be set automatically (default: None).
    num_cols : int, optional
        Number of columns of subplots, can be set automatically (default: None).
    layout : Plotly layout, optional
        A layout for the figure, overrides the layout options (default: None).
    layout_options : dict, optional
        Settings to modify the default layout (default: DEFAULT_LAYOUT_OPTIONS).
    trace_options : dict, optional
        Settings to modify the default trace type (default: DEFAULT_TRACE_OPTIONS).
    trace_names : str, optional
        Name(s) for the primary trace(s) (default: None).
    trace_name_width : int, optional
        Maximum length of the trace names before text wrapping is used (default: 40).

    Returns
    -------
    plotly.graph_objs.Figure
        The generated Plotly figure.
    """

    def __init__(
        self,
        x,
        y,
        num_rows=None,
        num_cols=None,
        axis_titles=None,
        layout=None,
        layout_options=DEFAULT_LAYOUT_OPTIONS,
        subplot_options=DEFAULT_SUBPLOT_OPTIONS,
        trace_options=DEFAULT_SUBPLOT_TRACE_OPTIONS,
        trace_names=None,
        trace_name_width=40,
    ):
        super().__init__(
            x, y, layout, layout_options, trace_options, trace_names, trace_name_width
        )
        self.num_traces = len(self.traces)
        self.num_rows = num_rows
        self.num_cols = num_cols
        if self.num_rows is None and self.num_cols is None:
            # Work out the number of subplots
            self.num_cols = int(math.ceil(math.sqrt(self.num_traces)))
            self.num_rows = int(math.ceil(self.num_traces / self.num_cols))
        elif self.num_rows is None:
            self.num_rows = int(math.ceil(self.num_traces / self.num_cols))
        elif self.num_cols is None:
            self.num_cols = int(math.ceil(self.num_traces / self.num_rows))
        self.axis_titles = axis_titles
        self.subplot_options = subplot_options.copy()
        if subplot_options is not None:
            for arg, value in subplot_options.items():
                self.subplot_options[arg] = value

        # Attempt to import plotly when an instance is created
        self.make_subplots = PlotlyManager().make_subplots

    def __call__(self, show):
        """
        Generate and show the set of figures.

        Parameters
        ----------
        show : bool, optional
            If True, the figure is shown upon creation (default: True).
        """
        fig = self.make_subplots(
            rows=self.num_rows,
            cols=self.num_cols,
            horizontal_spacing=0.1,
            vertical_spacing=0.15,
            **self.subplot_options,
        )
        fig.update_layout(self.layout_options)

        for idx, trace in enumerate(self.traces):
            row = (idx // self.num_cols) + 1
            col = (idx % self.num_cols) + 1
            fig.add_trace(trace, row=row, col=col)

            if self.axis_titles and idx < len(self.axis_titles):
                x_title, y_title = self.axis_titles[idx]
                fig.update_xaxes(title_text=x_title, row=row, col=col)
                fig.update_yaxes(
                    title_text=y_title,
                    row=row,
                    col=col,
                    showexponent="last",
                    exponentformat="e",
                )

        if show:
            fig.show()

        return fig


def plot_trajectories(x, y, trace_names=None, show=True, **layout_kwargs):
    """
    Quickly plot one or more trajectories using Plotly.

    Parameters
    ----------
    x : list or np.ndarray
        X-axis data points.
    y : list or np.ndarray
        Y-axis data points for each trajectory.
    trace_names : list or str, optional
        Name(s) for the trace(s) (default: None).
    **layout_kwargs : optional
            Valid Plotly layout keys and their values,
            e.g. `xaxis_title="Time / s"` or
            `xaxis={"title": "Time / s", "titlefont_size": 18}`.

    Returns
    -------
    plotly.graph_objs.Figure
        The Plotly figure object for the scatter plot.
    """
    # Create a plotting dictionary
    plot_dict = StandardPlot(
        x=x,
        y=y,
        trace_names=trace_names,
    )

    # Generate the figure and update the layout
    fig = plot_dict(show=False)
    fig.update_layout(**layout_kwargs)
    if show:
        fig.show()

    return fig
