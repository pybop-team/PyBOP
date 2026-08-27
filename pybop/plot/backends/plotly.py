import numbers

import numpy as np

from pybop.plot.backends.base import PlotBackend
from pybop.plot.backends.plotly_manager import PlotlyManager
from pybop.plot.util import wrap_text

# Mapping from Matplotlib line styles to Plotly dash styles.
LINESTYLE_MAP = {
    "solid": "solid",
    "dashed": "dash",
    "dotted": "dot",
    "dashdot": "dashdot",
}

# Mapping from Matplotlib-style marker definitions to their Plotly
# equivalents.
MARKER_MAP = {"o": "circle", "P": "cross", "X": "x", ".": None}

# Translation between Matplotlib legend anchor keywords and Plotly
# anchor names.
ANCHOR_MAP = {
    "lower": "bottom",
    "upper": "top",
    "left": "left",
    "center": "center",
    "right": "right",
}


class PlotlyBackend(PlotBackend):
    """
    Plotly implementation of the PlotBackend interface.

    This backend converts backend-agnostic plot definitions into Plotly
    figures and traces, providing interactive visualisations while
    maintaining a common plotting API.
    """

    def __init__(self):
        """
        Initialise the Plotly backend and associated Plotly manager.
        """
        self.plotly_manager = PlotlyManager()

    def _figure_layout(self, style, figure_title):
        axis_layout = dict(
            title=dict(font={"size": 14}),
            showexponent="last",
            exponentformat="e",
            tickfont=dict(size=12),
        )
        return {
            "title": figure_title,
            "width": style.get("width"),
            "height": style.get("height"),
            "xaxis": axis_layout,
            "yaxis": axis_layout,
            "plot_bgcolor": style.get("bg_color"),
        }

    @staticmethod
    def _check_axis_input(axis, raise_error=True):
        """
        Validate that the axis input is a tuple of two numbers.

        Parameters
        ----------
        axis : tuple
            Axis input to validate.

        Raises
        ------
        ValueError
            If the axis input is not a tuple of two numbers.
        """
        if (axis is not None) and (
            not isinstance(axis, (tuple, np.ndarray))
            or len(axis) != 2
            or not all(isinstance(x, numbers.Number) for x in axis)
        ):
            if raise_error:
                raise ValueError(
                    "Axis must be a tuple of the form (row, col) with numeric values."
                )
            else:
                return False
        return True

    def create_figure(
        self,
        title: str = None,
        xaxis_title: str = None,
        yaxis_title: str = None,
        traces=None,
        style: dict = None,
    ):
        """
        Create a Plotly figure.

        Parameters
        ----------
        title : str, optional
            Figure title.
        xaxis_title : str, optional
            X-axis label.
        yaxis_title : str, optional
            Y-axis label.
        traces : list, optional
            Plotly traces to add to the figure.
        style : dict, optional
            Currently supported options:
                - width in pixels
                - height in pixels
                - xaxis_range: range of the X-axis
                - yaxis_range: range of the Y-axis
                - bg_color: background color of the axis

        Returns
        -------
        plotly.graph_objects.Figure
            Configured Plotly figure.
        """
        style = style or {}
        layout_opts = self._figure_layout(style, title)
        layout_opts.update(
            {
                "xaxis_title": xaxis_title,
                "yaxis_title": yaxis_title,
                "xaxis_range": style.get("xaxis_range"),
                "yaxis_range": style.get("yaxis_range"),
                "barmode": "overlay",
            }
        )
        layout = self.plotly_manager.go.Layout(layout_opts)

        fig = self.plotly_manager.go.Figure(data=traces, layout=layout)
        return fig

    def make_subplots(
        self,
        num_rows: int,
        num_cols: int,
        num_plots: int,
        title=None,
        style=None,
    ):
        """
        Create a figure containing multiple subplots.

        Parameters
        ----------
        num_rows : int
            Number of rows in the subplot grid.
        num_cols : int
            Number of columns in the subplot grid.
        num_plots : int
            Total number of subplots to create.
        title : str, optional
            Figure title.
        xaxis_titles : str or list[str], optional
            X-axis titles.
        yaxis_titles : str or list[str], optional
            Y-axis titles.
        style : dict, optional
            Figure styling options.

        Returns
        -------
        tuple
            (
                figure,
                axes dictionary,
                number of rows,
                number of columns
            )
        """
        style = style or {}

        if (num_rows * num_cols) < num_plots:
            raise ValueError(
                f"Insufficient subplots: {num_rows} rows and {num_cols} columns "
                f"cannot accommodate {num_plots} plots."
            )

        # Create figure with supbplots
        make_subplots = self.plotly_manager.make_subplots
        fig = make_subplots(
            rows=num_rows,
            cols=num_cols,
            horizontal_spacing=0.2,
            vertical_spacing=0.15,
        )

        axes = [
            (row, col)
            for row in range(1, num_rows + 1)
            for col in range(1, num_cols + 1)
        ]

        fig.update_layout(self._figure_layout(style, title))

        return fig, axes[:num_plots]

    def legend(self, fig, axes=None, style: dict = None):
        """
        Configure and display a figure legend.

        Parameters
        ----------
        fig : plotly.graph_objects.Figure
            Target figure.
        style : dict, optional
            Legend styling options including orientation,
            location and anchor coordinates.
        axes : tuple, optional
            Subplot axis to apply the legend to (if applicable).
        """
        style = style or {}
        opts = {}
        if style.get("horizontal"):
            opts["orientation"] = "h"
        if "loc" in style:
            anchors = style.get("loc").split(" ")
            if len(anchors) != 2:
                raise ValueError("loc property must consist of 2 keywords")
            opts["xanchor"] = ANCHOR_MAP.get(anchors[1], "auto")
            opts["yanchor"] = ANCHOR_MAP.get(anchors[0], "auto")
        if "coords" in style:
            coords = style.get("coords")
            opts["x"] = coords[0]
            opts["y"] = coords[1]

        # Ignore options if axes is not None
        if axes is not None:
            opts = {}

        fig.update_layout(showlegend=True, legend=opts)

    def show_figure(self, fig):
        """
        Display one or more Plotly figures.

        Parameters
        ----------
        fig : Figure or iterable[Figure]
            Figure or collection of figures to display.
        """
        # Support displaying either a single figure or a collection of figures.
        if hasattr(fig, "__len__") and len(fig) > 0:
            # show duplicate figures only once
            seen = set()
            for f in fig:
                if id(f) not in seen:
                    seen.add(id(f))
                    f.show()
        else:
            fig.show()

    def parse_input_axes(self, figures, axes, num_plots=None, allow_single_axis=True):
        """
        Parse and validate the input axes for plotting.

        Parameters
        ----------
        figures : Figure or list[Figure]
            Plotly figure(s) to which the axes belong.
        axes : tuple or list[tuple]
            Subplot locations specified as (row, col) tuples.
        num_plots : int, optional
            Number of plots to be displayed.
        allow_single_axis : bool, optional
            If True, a single axis can be used for all plots.

        Returns
        -------
        tuple
            (
                list of figures,
                list of axes,
                number of rows,
                number of columns
            )
        Raises
        """
        if axes is not None:
            if self._check_axis_input(axes, raise_error=False):
                axes = [axes]
            else:
                axes = list(axes)
                for ax in axes:
                    self._check_axis_input(ax)
        return super().parse_input_axes(
            figures, axes, num_plots, allow_single_axis=allow_single_axis
        )

    def update_axes_titles(
        self, figures, axes, xaxis_titles, yaxis_titles, max_width=40
    ):
        """
        Update the titles of the axes in the provided figures.

        Parameters
        ----------
        figures : list[Figure]
            List of plotly figures containing the axes to update.
        axes : list[tuple]
            List of subplot locations specified as (row, col) tuples.
        xaxis_titles : list[str]
            List of titles for the X-axes.
        yaxis_titles : list[str]
            List of titles for the Y-axes.
        max_width : int, optional
            Maximum width for the axis titles before wrapping. Default is 40 characters.
        """

        figures, axes, _, _ = self.parse_input_axes(
            figures, axes, allow_single_axis=True
        )
        xaxis_titles = np.atleast_1d(xaxis_titles)
        yaxis_titles = np.atleast_1d(yaxis_titles)
        for i, ax in enumerate(axes):
            # Wrap the axis titles to the specified maximum width
            xaxis_title = xaxis_titles[i % len(xaxis_titles)]
            if xaxis_title is not None:
                xaxis_title = wrap_text(xaxis_title, width=max_width)
            yaxis_title = yaxis_titles[i % len(yaxis_titles)]
            if yaxis_title is not None:
                yaxis_title = wrap_text(yaxis_title, width=max_width)
            if ax is None:
                figures[i].update_layout(xaxis_title=xaxis_title)
                figures[i].update_layout(yaxis_title=yaxis_title)
            else:
                figures[i].update_xaxes(
                    title_text=xaxis_title,
                    row=ax[0],
                    col=ax[1],
                )
                figures[i].update_yaxes(
                    title_text=yaxis_title,
                    row=ax[0],
                    col=ax[1],
                )

    def update_plot_titles(self, figures, axes, titles, max_text_width=40, pad=0):
        """
        Update the titles of the subplots in the provided figures.

        Parameters
        ----------
        figures : list[Figure]
            List of plotly figures containing the subplots to update.
        axes : list[tuple]
            List of subplot locations specified as (row, col) tuples.
        titles : list[str]
            List of titles for the subplots.
        max_text_width : int, optional
            Maximum width for the subplot titles before wrapping. Default is 40 characters.
        pad : int, optional
            Padding between the title and the subplot. Default is 0.
            Ignored for plotly. Added for consistency with matplotlib backend.

        """
        titles = np.atleast_1d(titles)
        figures, axes, _, _ = self.parse_input_axes(
            figures, axes, allow_single_axis=True
        )
        for i, ax in enumerate(axes):
            title = titles[i % len(titles)]
            if title is not None:
                title = wrap_text(title, width=max_text_width)
            if title is None:
                continue
            if ax is None:
                figures[i].update_layout(title=title)
            else:
                figures[i].add_annotation(
                    xref="x domain",
                    yref="y domain",
                    x=0.0,
                    y=1.05,
                    showarrow=False,
                    text=title,
                    row=ax[0],
                    col=ax[1],
                    font=dict(size=14),
                )

    def equal_aspect(self, fig, ax=None):
        """
        Constrain the axes of a subplot to an equal aspect ratio.

        Parameters
        ----------
        fig : plotly.graph_objects.Figure
            Figure containing the axes to update.
        ax : tuple, optional
            Subplot location.
        """
        if ax is None:
            fig.update_yaxes(scaleanchor="x", scaleratio=1)
        else:
            self._check_axis_input(ax)
            # Each subplot anchors to its own x-axis, e.g. "x5"
            fig.update_yaxes(
                scaleanchor=fig.get_subplot(*ax).yaxis.anchor,
                scaleratio=1,
                row=ax[0],
                col=ax[1],
            )

    def update_axes_ranges(self, fig, ax, xaxis_range, yaxis_range):
        """
        Update the ranges of the axes in the provided figure.

        Parameters
        ----------
        fig : Figure
            Plotly figure containing the axes to update.
        ax : tuple
            Subplot location.
        xaxis_range : tuple
            Range for the x-axis.
        yaxis_range : tuple
            Range for the y-axis.
        """
        if ax is None:
            fig.update_layout(xaxis_range=xaxis_range, yaxis_range=yaxis_range)
        else:
            self._check_axis_input(ax)
            fig.update_xaxes(range=xaxis_range, row=ax[0], col=ax[1])
            fig.update_yaxes(range=yaxis_range, row=ax[0], col=ax[1])

    def plot_trace(self, traces, fig, ax=None):
        """
        Add one or more traces to a figure or subplot.

        Parameters
        ----------
        traces : Trace or list[Trace]
            Plotly trace objects to add.
        fig : plotly.graph_objects.Figure
            Target figure.
        ax : tuple, optional
            Subplot location. If provided, traces are added
            to the specified subplot.

        Returns
        -------
        None
        """
        for trace in np.atleast_1d(traces):
            if isinstance(trace, self.plotly_manager.go.Histogram):
                # Use barmode='overlay' for histograms
                fig.update_layout(barmode="overlay")
            if ax is None:
                fig.add_trace(trace)
            else:
                self._check_axis_input(ax)
                fig.add_trace(trace, row=ax[0], col=ax[1])

    def sample_color_scale(self, data, scale="viridis", d_min=None, d_max=None):
        """
        Sample colours from a Plotly colour scale.

        Parameters
        ----------
        data : array-like
            Values to map onto the colour scale.
        scale : str, optional
            Plotly colour scale name.
        d_min : float, optional
            Minimum value used for normalisation.
        d_max : float, optional
            Maximum value used for normalisation.

        Returns
        -------
        list
            Colours corresponding to the supplied values.
        """
        px = self.plotly_manager.px
        # normalise and clip data
        d_min = d_min or np.nanmin(data[np.isfinite(data)])
        d_max = d_max or np.nanmax(data[np.isfinite(data)])

        d = (data - d_min) / (d_max - d_min)
        if np.isscalar(d):
            d = np.array([d])
        np.clip(np.asarray(d), 0, 1.0, out=d)
        return px.colors.sample_colorscale(scale, list(d))

    def colorbar(self, fig, data, colorscale="viridis", label=None, ax=None):
        """
        Add a standalone colour bar to a figure.

        Parameters
        ----------
        fig : plotly.graph_objects.Figure
            Target figure.
        data : array-like
            Values defining the colour range.
        colorscale : str, optional
            Plotly colour scale name.
        label : str, optional
            Colour bar title.

        Returns
        -------
        None
        """
        d_min = np.nanmin(data[np.isfinite(data)])
        d_max = np.nanmax(data[np.isfinite(data)])

        colorbar = dict(thickness=25, outlinewidth=1)
        if label is not None:
            colorbar.update({"title": {"text": label, "side": "right"}})

        if ax is not None:
            self._check_axis_input(ax)
            xaxis = next(fig.select_xaxes(row=ax[0], col=ax[1]))
            yaxis = next(fig.select_yaxes(row=ax[0], col=ax[1]))
            colorbar.update(
                yanchor="bottom",
                y=yaxis.domain[0],
                x=xaxis.domain[1],
                len=yaxis.domain[1] - yaxis.domain[0],
            )

        # Plotly requires a trace to render a standalone colour bar, so an
        # invisible scatter trace is added solely to display the scale.
        trace = self.plotly_manager.go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(
                colorscale=colorscale,
                showscale=True,
                cmin=d_min,
                cmax=d_max,
                colorbar=colorbar,
            ),
            showlegend=False,
            hoverinfo="none",
        )
        self.plot_trace(trace, fig, ax)

    def contour_plot(self, x, y, z, fig, ax, colorscale="viridis"):
        """
        Create a contour plot trace.

        Parameters
        ----------
        x, y : array-like
            Coordinate values.
        z : array-like
            Contour values.
        colorscale : str, optional
            Plotly colour scale.
        fig : plotly.graph_objects.Figure
            Target figure.
        ax : tuple, optional
            Subplot location. If provided, the contour trace is added to the specified subplot.

        Returns
        -------
        plotly.graph_objects.Contour
            Contour trace.
        """
        colorbar = {}
        if ax is not None:
            self._check_axis_input(ax)
            xaxis = next(fig.select_xaxes(row=ax[0], col=ax[1]))
            yaxis = next(fig.select_yaxes(row=ax[0], col=ax[1]))
            colorbar.update(
                yanchor="bottom",
                y=yaxis.domain[0],
                x=xaxis.domain[1],
                len=yaxis.domain[1] - yaxis.domain[0],
            )

        # Use connectgaps=True to ill small gaps in the input grid to avoid breaks in contour regions.
        trace = self.plotly_manager.go.Contour(
            x=x, y=y, z=z, colorscale=colorscale, connectgaps=True, colorbar=colorbar
        )
        self.plot_trace(trace, fig, ax=ax)

    def fill_between(self, x, y_upper, y_lower, color):
        """
        Create a filled region between two curves.

        Parameters
        ----------
        x : array-like
            X values.
        y_upper : array-like
            Upper boundary.
        y_lower : array-like
            Lower boundary.
        color : str
            Fill colour.

        Returns
        -------
        plotly.graph_objects.Scatter
            Filled area trace.
        """

        # Construct a closed polygon by traversing the upper curve forwards
        # and the lower curve in reverse.
        return self.plotly_manager.go.Scatter(
            x=x + x[::-1],
            y=y_upper + y_lower[::-1],
            fill="toself",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=False,
            fillcolor=color,
        )

    def heatmap(self, x, y, z, colorscale="viridis"):
        """
        Create a heatmap trace.

        Parameters
        ----------
        x, y : array-like
            Coordinate values.
        z : array-like
            Heatmap values.
        colorscale : str, optional
            Plotly colour scale.

        Returns
        -------
        plotly.graph_objects.Heatmap
            Heatmap trace.
        """
        return self.plotly_manager.go.Heatmap(
            x=x, y=y, z=z, colorscale=colorscale, zsmooth="best", showscale=False
        )

    def histogram_plot(self, x, name, style=None):
        """
        Create a histogram trace.

        Parameters
        ----------
        x : array-like
            Data values.
        name : str
            Histogram label.
        style : dict, optional
            Histogram styling options.

        Returns
        -------
        plotly.graph_objects.Histogram
            Histogram trace.
        """
        style = style or {}

        return self.plotly_manager.go.Histogram(
            x=x, name=name, opacity=style.get("alpha")
        )

    def _get_line_options(self, style, opts):
        """
        Populate Plotly line styling options.

        Parameters
        ----------
        style : dict
            User-supplied style options.
        opts : dict
            Trace options dictionary updated in-place.

        Returns
        -------
        None
        """
        linestyle = style.get("linestyle", "solid")
        color = style.get("color")
        opts["line"] = dict(
            width=style.get("linewidth", 4), dash=LINESTYLE_MAP.get(linestyle, "solid")
        )
        if color is not None:
            opts["line"].update(color=color)

    def line(self, x=None, y=None, label=None, style=None):
        """
        Create a line and/or marker trace.

        Parameters
        ----------
        x : array-like, optional
            X values.
        y : array-like
            Y values.
        label : str, optional
            Legend label.
        style : dict, optional
            Line and marker styling options.
            Currently supported:
                - linestyle: see LINESTYLE_MAP
                - linewidth
                - color
                - marker: the marker symbol see MARKER_MAP
                - markerfacecolor
                - markeredgecolor
                - markeredgewidth
                - fillstyle (for marker)

        Returns
        -------
        plotly.graph_objects.Scatter
            Scatter trace configured as a line, marker plot,
            or combined line-marker plot.
        """

        style = style or {}

        if y is None:
            raise ValueError("y must be provided")

        linestyle = style.get("linestyle", "solid")
        marker = style.get("marker", "none")
        opts = {}
        if linestyle.lower() == "none":
            mode = "markers"
        elif marker.lower() == "none":
            mode = "lines"
        else:
            mode = "markers+lines"

        opts["mode"] = mode
        if linestyle.lower() != "none":
            self._get_line_options(style, opts)

        if marker.lower() != "none":
            opts["marker"] = dict(
                size=style.get("markersize", 8),
                symbol=MARKER_MAP.get(marker),
            )
            fillstyle = style.get("fillstyle", "full")
            markerfacecolor = style.get("markerfacecolor")
            markeredgecolor = style.get("markeredgecolor")
            markeredgewidth = style.get("markeredgewidth")

            # Plotly uses "-open" marker variants to represent unfilled markers.
            if fillstyle.lower() == "none":
                opts["marker"].update(symbol=MARKER_MAP.get(marker) + "-open")
                if markeredgecolor is not None:
                    opts["marker"].update(color=markeredgecolor)

            if markerfacecolor is not None:
                opts["marker"].update(color=markerfacecolor)
            if markeredgecolor is not None:
                opts["marker"].update(
                    line_color=markeredgecolor, line_width=markeredgewidth or 1
                )
            elif markeredgewidth is not None:
                opts["marker"].update(line_width=markeredgewidth)

        # Avoid creating empty legend entries for unnamed traces.
        if label is None:
            opts["showlegend"] = False

        kwargs = {"y": y, "name": label, **opts}
        if x is not None:
            kwargs["x"] = x

        return self.plotly_manager.go.Scatter(**kwargs)

    def scatter(self, x, y, colors, labels=None, colorscale="Greys"):
        """
        Create a scatter plot trace.

        Parameters
        ----------
        x, y : array-like
            Point coordinates.
        colors : array-like
            Values used to colour markers.
        labels : array-like, optional
            Hover labels.
        colorscale : str, optional
            Plotly colour scale.

        Returns
        -------
        plotly.graph_objects.Scatter
            Scatter trace.
        """

        opts = dict(
            mode="markers",
            marker=dict(
                color=colors,
                colorscale=colorscale,
                size=8,
                showscale=False,
            ),
            showlegend=False,
        )
        if labels is not None:
            opts.update({"text": labels, "hoverinfo": "text"})
        return self.plotly_manager.go.Scatter(x=x, y=y, **opts)

    def show_table(self, header, values, title, fig=None, ax=None):
        """
        Display tabular data as a Plotly table.

        Parameters
        ----------
        header : list
            Column headers.
        values : list
            Table contents.
        title : str
            Table title.
        fig : plotly.graph_objects.Figure, optional
            Figure for plotting. If not provided a new figure is created.
        ax : tuple, optional
            Subplot location. If provided, the table is added to the specified subplot.

        Returns
        -------
        None

        NOTE: There seems to be a bug in plotly https://github.com/plotly/plotly.py/issues/3424
        If any plot with a vline is added to a figure with subplots AFTER a table was added,
        this may cause an error. This can be avoided by always adding any tables last.
        """
        fig = fig or self.plotly_manager.go.Figure()

        if ax is not None:
            # Retrieve subplot domain
            self._check_axis_input(ax)
            domain = fig.get_subplot(ax[0], ax[1])
            x1, x2 = domain.x
            y1, y2 = domain.y

            # Add title as annotation within subplot domain
            fig.add_annotation(
                xref="paper",
                yref="paper",
                xanchor="left",
                x=x1,
                y=y1 + 0.9 * (y2 - y1),
                text=title,
                showarrow=False,
                font=dict(size=14),
            )

            # Manually position table in subplot domain
            # This is to enable creating space for the title above the table
            trace = self.plotly_manager.go.Table(
                domain=dict(
                    x=[x1, x2],
                    y=[y1, y1 + 0.8 * (y2 - y1)],  # leaves 20% above
                ),
                header=dict(values=header),
                cells=dict(
                    values=[[row[0] for row in values], [row[1] for row in values]]
                ),
            )
        else:
            fig.update_layout(title=title)
            trace = self.plotly_manager.go.Table(
                header=dict(values=header),
                cells=dict(
                    values=[[row[0] for row in values], [row[1] for row in values]]
                ),
            )

        self.plot_trace(trace, fig)

        return fig

    def vline(self, fig, x, style=None, ax=None):
        """
        Add a vertical reference line to a figure.

        Parameters
        ----------
        fig : plotly.graph_objects.Figure
            Target figure.
        x : float
            X-coordinate of the line.
        style : dict, optional
            Line styling options.
        ax : tuple, optional
            Subplot location. If provided, the line is added to the specified subplot.

        Returns
        -------
        None

        NOTE: There seems to be a bug in plotly https://github.com/plotly/plotly.py/issues/3424
            If any plot with a vline is added to a figure with subplots AFTER a table was added
            (e.g. using show_table), this may cause an error.
            This can be avoided by always adding any tables last.
        """
        style = style or {}
        opts = {}
        self._get_line_options(style, opts)
        if ax is not None:
            self._check_axis_input(ax)
            fig.add_vline(x=x, **opts, row=ax[0], col=ax[1])
        else:
            fig.add_vline(x=x, **opts)
