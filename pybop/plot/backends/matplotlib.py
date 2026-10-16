import numpy as np

from pybop.plot.backends.base import PlotBackend
from pybop.plot.util import wrap_text


class MatplotlibBackend(PlotBackend):
    """
    Matplotlib implementation of the PlotBackend interface.
    This backend converts backend-agnostic trace definitions into
    Matplotlib figures, axes, and artists. Plot objects are represented
    as dictionaries containing plotting arguments and metadata, allowing
    higher-level plotting code to remain independent of the underlying
    plotting library.
    """

    def __init__(self):
        # Import matplotlib only when needed
        import matplotlib as mpl
        from matplotlib import pyplot as plt

        self.mpl = mpl
        self.plt = plt

        # Enable automatic colour cycling across subplots when traces do not
        # explicitly define a colour.
        self.global_colorcycle = False

        # Matplotlib's default property cycle.
        self.colorcycle = self.plt.rcParams["axes.prop_cycle"]()

        # Layout rectangle reserved for tight_layout(). This may be adjusted
        # when legends are placed outside the plotting area.
        self.rect = [0, 0, 1, 1]

    def _figsize(self, style):
        """
        Convert pixel-based width and height values from a style dictionary
        into a Matplotlib figsize (inches).
        """
        if "width" in style or "height" in style:
            return (
                np.ceil(style.get("width", 800) / 100),
                np.ceil(style.get("height", 600) / 100),
            )
        else:
            return None

    def create_figure(
        self, title=None, xaxis_title=None, yaxis_title=None, traces=None, style=None
    ):
        """
        Create a single-axis figure and optionally populate it with traces.

        Parameters
        ----------
        title : str, optional
            Figure title.
        xaxis_title : str, optional
            X-axis label.
        yaxis_title : str, optional
            Y-axis label.
        traces : list[dict], optional
            Trace definitions to plot immediately.
        style : dict, optional
            Figure styling options.
            Currently supported options:
                - width in pixels
                - heith in pixels
                - xaxis_range: range of the X-axis
                - yaxis_range: range of the Y-axis
                - bg_color: background color of the axis

        Returns
        -------
        matplotlib.figure.Figure
            Configured figure instance.
        """
        style = style or {}
        fig = self.plt.figure(figsize=self._figsize(style))

        if title is not None:
            self.plt.suptitle(title)
        if xaxis_title is not None:
            self.plt.xlabel(xaxis_title)
        if yaxis_title is not None:
            self.plt.ylabel(yaxis_title)

        # Apply backend-supported figure styling options.
        self.update_axes_ranges(
            fig,
            xaxis_range=style.get("xaxis_range"),
            yaxis_range=style.get("yaxis_range"),
        )
        if "bg_color" in style:
            ax = fig.gca()
            ax.set_facecolor(style.get("bg_color"))
            ax.set_axisbelow(True)

        if traces is not None:
            for trace in traces:
                self.plot_trace(trace, fig)
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
        Create a figure containing multiple subplot axes in a grid.

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
        style : dict, optional
            Figure styling options.
            Currently supported options:
                - width in pixels
                - heith in pixels
                - bg_color: background color of the axis

        Returns
        -------
        matplotlib.figure.Figure
            Configured figure instance.
        """

        style = style or {}
        if (num_rows * num_cols) < num_plots:
            raise ValueError(
                f"Insufficient subplots: {num_rows} rows and {num_cols} columns "
                f"cannot accommodate {num_plots} plots."
            )

        # Create figure
        fig, axes = self.plt.subplots(
            num_rows, num_cols, figsize=self._figsize(style), dpi=100
        )
        axes = np.atleast_1d(axes).flatten()
        for ax in axes[num_plots:]:
            ax.set_visible(False)

        if title is not None:
            self.plt.suptitle(title)

        for ax in fig.axes:
            if "bg_color" in style:
                ax.set_facecolor(style.get("bg_color"))
                ax.set_axisbelow(True)

        # Use a shared colour cycle across all subplot axes.
        self.global_colorcycle = True
        return fig, axes[:num_plots]

    def legend(self, fig, axes=None, style: dict = None):
        """
        Create an axis-level or figure-level legend.

        Supports legends positioned outside the plotting area and updates
        the layout rectangle used by tight_layout() accordingly.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure object
        style : dict, optional
            Legend styling options.
            Currently supported options:
                - loc: str
                - coords: tuple - is translated into bbox_to_anchor
                - outside: tuple(side : str, offset: float) places
                    the legend outside the plot, where the side (left,
                    right, top, bottom) determines on wich side of the plot
                    the legend is placed and the offset determines the fraction
                    of the figure height or width reserved for the legend.
                    Overrides loc and coords.
                - fig_legend: if true, one legend is created for the entire figure, otherwise the legend is created
                    for the current axis.

        """
        style = style or {}
        lines_labels = []
        if style.get("fig_legend"):
            axes = fig.axes
        elif axes is not None:
            axes = np.atleast_1d(axes)
        else:
            axes = [fig.gca()]

        if "loc" in style:
            anchors = style.get("loc").split(" ")
            if len(anchors) != 2:
                raise ValueError("loc property must consist of 2 keywords")

        # Configure external legend placement and reserve layout space.
        if "outside" in style.keys():
            side, offset = style.get("outside")
            if side == "left":
                style["loc"] = "upper left"
                style["coords"] = (0.0, 1.0)
                self.rect = [offset, 0, 1, 1]
            elif side == "top":
                style["loc"] = "lower right"
                style["coords"] = (1.0, 1.0 - offset)
                self.rect = [0, 0, 1, 1 - offset]
            elif side == "bottom":
                style["loc"] = "lower left"
                style["coords"] = (0.0, 0.0)
                self.rect = [0, offset, 1, 1]
            else:
                style["loc"] = "upper right"
                style["coords"] = (1.0, 1.0)
                self.rect = [0.0, 0, 1 - offset, 1]

        # Collect legend entries from all relevant axes.
        labels_in_fig = False
        lines_labels = []
        opts = {}
        for ax in axes:
            # Flatten handles and labels from multiple axes into a single legend.
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                lines_labels.append((handles, labels))
                labels_in_fig = True

        if labels_in_fig:
            lines, labels = [sum(lol, []) for lol in zip(*lines_labels, strict=False)]
            if style.get("horizontal"):
                opts["ncols"] = len(lines)
            if "coords" in style.keys():
                opts["bbox_to_anchor"] = style.get("coords")
            if style.get("fig_legend"):
                opts["loc"] = style.get("loc", "upper right")
                fig.legend(lines, labels, **opts)
            else:
                opts["loc"] = style.get("loc", "best")
                axes[-1].legend(lines, labels, **opts)

    def update_axes_titles(
        self, figures, axes, xaxis_titles, yaxis_titles, max_width=40
    ):
        """
        Update the titles of the axes in the provided figures.

        Parameters
        ----------
        figures : list[Figure]
            List of matplotlib figures containing the axes to update.
        axes : list[tuple]
            List of subplot locations specified as matplotlib axes objects.
        xaxis_titles : list[str]
            List of titles for the X-axes.
        yaxis_titles : list[str]
            List of titles for the Y-axes.
        max_width : int, optional
            Maximum width for the axis titles before wrapping. Default is 40 characters.
        """
        xaxis_titles = np.atleast_1d(xaxis_titles)
        yaxis_titles = np.atleast_1d(yaxis_titles)
        figures = np.atleast_1d(figures)

        for i, ax in enumerate(np.atleast_1d(axes)):
            if ax is None:
                ax = figures[i % len(figures)].gca()
            xaxis_title = xaxis_titles[i % len(xaxis_titles)]
            if xaxis_title is not None:
                xaxis_title = wrap_text(xaxis_title, width=max_width)
            ax.set_xlabel(xaxis_title)
            yaxis_title = yaxis_titles[i % len(yaxis_titles)]
            if yaxis_title is not None:
                yaxis_title = wrap_text(yaxis_title, width=max_width)
            ax.set_ylabel(yaxis_title)

    def update_plot_titles(self, figures, axes, titles, max_text_width=40, pad=0):
        """
        Update the titles of the subplots in the provided figures.

        Parameters
        ----------
        figures : list[Figure]
            List of matplotlib figures containing the subplots to update.
        axes : list[tuple]
            List of subplot locations specified as matplotlib axes objects.
        titles : list[str]
            List of titles for the subplots.
        max_text_width : int, optional
            Maximum width for the subplot titles before wrapping. Default is 40 characters.
        pad : int, optional
            Padding between the title and the subplot. Default is 0.
        """
        titles = np.atleast_1d(titles)
        figures = np.atleast_1d(figures)
        for i, ax in enumerate(np.atleast_1d(axes)):
            if ax is None:
                ax = figures[i].gca()
            title = titles[i % len(titles)]
            if title is not None:
                title = wrap_text(title, width=max_text_width)
            ax.set_title(title, pad=pad)

    def equal_aspect(self, fig, ax=None):
        """
        Constrain the axes of a subplot to an equal aspect ratio.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure containing the axes to update.
        ax : matplotlib.axes.Axes, optional
            The axes to update. Defaults to the current axes of the figure.
        """
        ax = ax if ax is not None else fig.gca()
        ax.set_aspect("equal", adjustable="datalim")

    def update_axes_ranges(self, fig, ax=None, xaxis_range=None, yaxis_range=None):
        """
        Update the ranges of the axes in the provided figure.

        Parameters
        ----------
        fig : Figure
            Matplotlib figure containing the axes to update.
        ax : tuple
            Subplot location.
        xaxis_range : tuple
            Range for the x-axis.
        yaxis_range : tuple
            Range for the y-axis.
        """
        ax = ax or fig.gca()
        if xaxis_range is not None:
            ax.set_xlim(xaxis_range)
        if yaxis_range is not None:
            ax.set_ylim(yaxis_range)

    def show_figure(self, fig):
        """
        Apply final layout adjustments and display the figure.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure object
        """

        if isinstance(fig, (list, np.ndarray, tuple)):
            for f in fig:
                f.tight_layout(rect=self.rect)
        else:
            fig.tight_layout(rect=self.rect)
        self.plt.show()

    def plot_trace(self, traces: dict | list[dict], fig, ax=None):
        """
        Convert one or more trace definitions into Matplotlib plotting calls.

        Parameters
        ----------
        traces: dict or list[dict]
            Each trace dictionary specifies a plotting method, positional
            arguments, and keyword arguments compatible with a Matplotlib Axes
            method.
        fig : matplotlib.figure.Figure
            The figure object
        ax : matplotlib axis object, optional
            Specity an axis for plotting. Otherwise current axis is used.
        """

        traces = traces if isinstance(traces, list) else [traces]

        # Extract plotting keyword arguments while removing backend metadata.
        ax = ax or fig.gca()
        fig.sca(ax)
        for trace in traces:
            if title := trace.get("xaxis_title"):
                ax.set_xlabel(title)
            if title := trace.get("yaxis_title"):
                ax.set_ylabel(title)

            options = {
                k: v
                for k, v in trace.items()
                if k
                not in {
                    "plot_type",
                    "positional_args",
                    "xaxis_title",
                    "yaxis_title",
                }
            }
            plot_type = trace.get("plot_type", "plot")
            args = trace.get("positional_args", ())

            # Apply the global colour cycle when plotting standard line traces.
            if (
                self.global_colorcycle
                and plot_type == "plot"
                and "color" not in options
                and "markeredgecolor" not in options
                and "markerfacecolor" not in options
            ):
                options.update(next(self.colorcycle))
            # Resolve the requested plotting method on the target axis.
            plot_func = getattr(ax, plot_type)

            obj = plot_func(*args, **options)

            # Automatically attach a colourbar for filled contour plots.
            if plot_type == "contourf":
                self.plt.colorbar(obj)

    def sample_color_scale(self, data, scale="viridis", d_min=None, d_max=None):
        """
        Map data values to RGBA colours using a Matplotlib colormap.

        Parameters
        ----------
        data: ndarray
            The data to be mapped
        scale : str
            Name of the colormap
        d_min: float, optional
            Minimum value to be mapped. Otherwise the minimum of the data is used.
        d_max: float, optional
            Maximum value to be mapped. Ohterwise maximum of the data is used.
        """
        # Normalise values into the range expected by the colormap.
        d_min = d_min or np.nanmin(data[np.isfinite(data)])
        d_max = d_max or np.nanmax(data[np.isfinite(data)])
        norm = self.mpl.colors.Normalize(vmin=d_min, vmax=d_max, clip=True)
        norm_d = norm(data, clip=True)
        if np.isscalar(norm_d):
            norm_d = [norm_d]

        # Sample colours from the requested colormap.
        cmap = self.mpl.colormaps[scale]
        return cmap(norm_d)

    def colorbar(self, fig, data, colorscale="viridis", label=None, ax=None):
        """
        Add colourbar to figure

        Parameters
        ----------
        fig: matplotlib.figure.Figure
            The figure.
        data: array-like
            The data to be mapped
        scale : str
            Name of the colormap
        label: str, optional
            label to be displayed alongside colorbar
        ax: matplotlib axis object, optional
            Specity an axis for plotting. Otherwise current axis is used.
        """
        # Get axis
        ax = ax or fig.gca()

        # Create a normalisation matching the supplied data range.
        f_min = np.nanmin(data[np.isfinite(data)])
        f_max = np.nanmax(data[np.isfinite(data)])
        norm = self.mpl.colors.Normalize(vmin=f_min, vmax=f_max, clip=True)

        # Use AxisDivider to adjust position and size of the colourbar relative to the target axis.
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        # Create and attach a standalone colourbar.
        cmap = self.mpl.colormaps[colorscale]
        self.plt.colorbar(
            self.mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=ax,
            label=label,
            cax=cax,
        )
        fig.sca(ax)

    def contour_plot(self, x, y, z, fig, ax=None, colorscale="viridis"):
        """
        Return trace definitions for a filled contour plot and contour lines.

        Parameters
        ----------
        x, y : array-like
            Coordinate values.
        z : array-like
            Surface values.
        colorscale : str, optional
            Colour scale name.
        fig : matplotlib.figure.Figure
            The figure.
        ax : matplotlib axis object, optional
            Specity an axis for plotting. Otherwise current axis is used.

        Returns
        -------
        object
            dictionary for contour plot definition and
            dictionary for contour line definition
        """
        contour = dict(positional_args=[x, y, z], plot_type="contourf", cmap=colorscale)
        contour_lines = dict(
            positional_args=[x, y, z],
            colors=("k"),
            linestyles="solid",
            linewidths=0.2,
            plot_type="contour",
        )
        self.plot_trace(contour, fig, ax=ax)
        self.plot_trace(contour_lines, fig, ax=ax)

    def fill_between(self, x, y_upper, y_lower, color):
        """
        Return a trace definition for a filled region between two curves.

        Parameters
        ----------
        x : array-like
            X-axis values.
        y_upper : array-like
            Upper boundary values.
        y_lower : array-like
            Lower boundary values.
        color : str
            Fill colour.
        """
        return {
            "positional_args": (x, y_upper, y_lower),
            "plot_type": "fill_between",
            "color": color,
        }

    def heatmap(self, x, y, z, colorscale="viridis"):
        """
        Return a trace definition for a heatmap.

        Parameters
        ----------
        x, y : array-like
            Coordinate values.
        z : array-like
            Surface values.
        colorscale : str, optional
            Colour scale name.
        """
        return {
            "positional_args": [z],
            "plot_type": "imshow",
            "cmap": colorscale,
            "extent": (np.min(x), np.max(x), np.min(y), np.max(y)),
            "origin": "lower",
            "interpolation": "nearest",
        }

    def histogram_plot(self, x, name, style: dict = None):
        """
        Return a trace definition for a histogram.

        Parameters
        ----------
        x : array-like
            Data to bin.
        name : str
            Histogram label.
        style : dict, optional
            Currently only 'alpha' supported for opacity.
            All other style arguments ignored.
        """
        style = style or {}

        return {
            "positional_args": [x],
            "label": name,
            "plot_type": "hist",
            "alpha": style.get("alpha"),
        }

    def line(self, x=None, y=None, label=None, style=None):
        """
        Return a trace definition for a line plot.

        Parameters
        ----------
        x, y : array-like, optional
            Coordinates of the line.
            If both x and y are provided, the shorter sequence determines the
            plotted length.
        label : str, optional
            Trace label.
        style : dict, optional
            Line styling options.

        Returns
        -------
        object
            dictionary with positional argumetns, label and style arguments
        """
        style = style or {}
        if y is None:
            raise ValueError("y must be provided")

        args = [y]
        if x is not None:
            size = min(len(x), len(y))
            args = [x[:size], y[:size]]

        return {
            "positional_args": args,
            "label": label,
            **style,
        }

    def scatter(self, x, y, colors=None, labels=None, colorscale="Greys"):
        """
        Return a trace definition for a scatter plot.

        Parameters
        ----------
        x, y : array-like
            Point coordinates.
        colors : array-like
            Values or colours associated with each point.
        labels : array-like, optional
            Point labels.
            Point labels are ignored by matplotlib implementation.
            Argument retained for consistency with plotly.
        colorscale : str, optional
            Colour scale name.
        """
        scatter = {
            "positional_args": [x, y],
            "plot_type": "scatter",
            "cmap": colorscale,
        }
        if colors is not None:
            scatter["c"] = colors
        return scatter

    def show_table(self, header, values, title, fig=None, ax=None):
        """
        Display tabular data in a standalone Matplotlib figure.

        Array-valued entries are converted to comma-separated strings before
        rendering.

        Parameters
        ----------
        header : list
            Column headers.
        values : list
            Table contents.
        title : str
            Table title.
        fig : matplotlib.figure.Figure, optional
            Figure for plotting. If not provided a new figure is created.
        ax : matplotlib axis object, optional
            Axis for plotting. If not provided the current axis is used.
        """
        for i, val in enumerate(values):
            values[i] = [val[0], ", ".join(val[1].astype(str))]

        if fig is None:
            fig = self.plt.figure(figsize=(6, 2), dpi=100)

        ax = ax or fig.gca()

        # Remove axis decorations so only the table is displayed.
        ax.axis("off")
        ax.axis("tight")
        ax.table(
            cellText=values,
            colLabels=header,
            loc="center",
            cellLoc="center",
            colColours=["lightsteelblue", "lightsteelblue"],
        )
        ax.set_title(title)
        return fig

    def vline(self, fig, x, style=None, ax=None):
        """
        Add a vertical reference line to the current axis.

        Parameters
        ----------
        fig: matplotlib.figure.Figure
            The figure.
        x: float
            The position of the vertical line on the axis
        style: dict, optional
            matplotlib arguments for axvline method
        ax: matplotlib axis object, optional
            Specity an axis for plotting. Otherwise current axis is used.
        """
        ax = ax or fig.gca()
        style = style or {}
        ax.axvline(x, **style)
