import warnings
from abc import ABC, abstractmethod

import numpy as np


class PlotBackend(ABC):
    """
    Abstract base class defining a plotting backend interface.

    Concrete implementations provide plotting functionality for a specific
    visualization library (e.g. Plotly, Matplotlib) while exposing a common
    API to the rest of the application.

    Methods in this interface are responsible for creating figures, adding
    traces and annotations, generating specialised plot types, and rendering
    results.
    """

    @property
    def name(self):
        """
        Return the name of the backend.

        Returns
        -------
        str
            Name of the backend.
        """
        return self.__class__.__name__.replace("Backend", "").lower()

    @abstractmethod
    def create_figure(
        self,
        title: str = None,
        xaxis_title: str = None,
        yaxis_title: str = None,
        traces: list = None,
        style: dict = None,
    ):
        """
        Create and return a new figure.

        Parameters
        ----------
        title : str, optional
            Figure title.
        xaxis_title : str, optional
            X-axis label.
        yaxis_title : str, optional
            Y-axis label.
        traces : list, optional
            Initial traces to add to the figure.
        style : dict, optional
            Backend-specific styling options.

        Returns
        -------
        object
            Backend-specific figure object.
        """
        raise NotImplementedError

    @abstractmethod
    def make_subplots(
        self,
        num_rows: int,
        num_cols: int,
        num_plots: int,
        title=None,
        style=None,
    ):
        """
        Create a figure containing multiple subplot axes.

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
            Backend-specific styling options.

        Returns
        -------
        fig : object
            Backend-specific figure object.
        axes : list
            List of subplot axes.
        """
        raise NotImplementedError

    @abstractmethod
    def legend(self, fig, ax=None, style: dict = None):
        """
        Configure or display a legend for a figure.

        Parameters
        ----------
        fig : object
            Figure object.
        style : dict, optional
            Legend styling options.
        ax : object, optional
            Subplot axis to apply the legend to (if applicable).
        """
        raise NotImplementedError

    @abstractmethod
    def show_figure(self, fig):
        """
        Render or display a figure.

        Parameters
        ----------
        fig : object
            Figure to display.
        """
        raise NotImplementedError

    def parse_input_axes(self, figures, axes, num_plots=None, allow_single_axis=True):
        """
        Parse and validate input figures and axes for plotting.

        Parameters
        ----------
        figures : object or list
            Figure(s) to plot on.
        axes : object or list
            Axis/axes to plot on.
        num_plots : int, optional
            Expected number of axes for the plot.
        allow_single_axis : bool, optional
            Whether to allow a single axis for multiple plots.

        Returns
        -------
        figures : list
            List of figure objects.
        axes : list
            List of axis objects.
        create_figure : bool
            Whether a new figure should be created.
        single_axis : bool
            Whether a single axis is being used for multiple plots.
        """
        if figures is None:
            if axes is not None:
                warnings.warn(
                    "Axes argument ignored if no figure provided.",
                    UserWarning,
                    stacklevel=2,
                )
            return [], [None], True, False

        figures = figures if hasattr(figures, "__len__") else [figures]
        axes = axes if hasattr(axes, "__len__") else [axes]
        if not len(figures) == len(axes):
            if len(figures) == 1:
                figures = np.asarray([figures[0]] * len(axes))
            elif axes[0] is None:
                axes = np.asarray([axes[0]] * len(figures))
            else:
                raise ValueError(
                    "Please provide the same number of figures and axes or only one figure."
                )
        if num_plots is not None and len(axes) != num_plots:
            if not allow_single_axis:
                raise ValueError(
                    f"This plot requires {num_plots} axes. {len(axes)} axes provided."
                )
            elif len(axes) != 1:
                raise ValueError(
                    f"This plot requires either {num_plots} axes or a single axis. {len(axes)} axes provided."
                )

        return figures, axes, False, len(axes) == 1

    @abstractmethod
    def update_axes_titles(self, figs, axes, xaxis_titles, yaxis_titles):
        """
        Update the titles of the axes in the provided figures.

        Parameters
        ----------
        figures : list[Figure]
            List of figures containing the axes to update.
        axes : list[tuple]
            List of subplot locations.
        xaxis_titles : list[str]
            List of titles for the X-axes.
        yaxis_titles : list[str]
            List of titles for the Y-axes.
        max_width : int, optional
            Maximum width for the axis titles before wrapping. Default is 40 characters.
        """
        raise NotImplementedError

    @abstractmethod
    def update_plot_titles(self, figs, axes, titles, pad):
        """
        Update the titles of the subplots in the provided figures.

        Parameters
        ----------
        figures : list[Figure]
            List of figures containing the subplots to update.
        axes : list[tuple]
            List of subplot locations.
        titles : list[str]
            List of titles for the subplots.
        max_text_width : int, optional
            Maximum width for the subplot titles before wrapping. Default is 40 characters.
        pad : int, optional
            Padding between the title and the subplot. Default is 0.
        """
        raise NotImplementedError

    @abstractmethod
    def equal_aspect(self, fig, ax=None):
        """
        Constrain the axes of a subplot to an equal aspect ratio, so that one unit is
        the same length on both axes. Used for e.g. Nyquist plots.

        Parameters
        ----------
        fig : Figure
            Figure containing the axes to update.
        ax : tuple, optional
            Subplot location.
        """
        raise NotImplementedError

    @abstractmethod
    def update_axes_ranges(self, fig, ax, xaxis_range, yaxis_range):
        """
        Update the ranges of the axes in the provided figure.

        Parameters
        ----------
        fig : Figure
            Figure containing the axes to update.
        ax : tuple
            Subplot location.
        xaxis_range : tuple
            Range for the x-axis.
        yaxis_range : tuple
            Range for the y-axis.
        """
        raise NotImplementedError

    @abstractmethod
    def plot_trace(self, traces: dict | list[dict], fig, ax=None, color_cycle=None):
        """
        Add one or more traces to a figure or subplot.

        Parameters
        ----------
        traces : dict or list[dict]
            Trace definitions to plot.
        fig : object
            Target figure.
        ax : object, optional
            Target subplot axis.
        color_cycle : iterable, optional
            Sequence of colours used when plotting multiple traces.
        """
        raise NotImplementedError

    @abstractmethod
    def sample_color_scale(self, data, scale="viridis", d_min=None, d_max=None):
        """
        Map data values onto a colour scale.

        Parameters
        ----------
        data : array-like
            Values to colour-map.
        scale : str, optional
            Colour scale name.
        d_min : float, optional
            Lower bound for normalisation.
        d_max : float, optional
            Upper bound for normalisation.

        Returns
        -------
        array-like
            Colours corresponding to the supplied data.
        """
        raise NotImplementedError

    @abstractmethod
    def colorbar(self, fig, data, colorscale="viridis", label=None):
        """
        Add a colour bar representing a colour scale.

        Parameters
        ----------
        fig : object
            Target figure.
        data : array-like
            Data used for colour scaling.
        colorscale : str, optional
            Colour scale name.
        label : str, optional
            Colour bar label.
        """
        raise NotImplementedError

    @abstractmethod
    def contour_plot(self, x, y, z, fig, ax=None, colorscale="viridis"):
        """
        Create a contour plot.

        Parameters
        ----------
        x, y : array-like
            Coordinate values.
        z : array-like
            Surface values.
        colorscale : str, optional
            Colour scale name.
        fig : object
            Target figure.
        ax : object, optional
            Target subplot axis.

        Returns
        -------
        object
            Backend-specific contour trace or figure.
        """
        raise NotImplementedError

    @abstractmethod
    def fill_between(self, x, y_upper, y_lower, color):
        """
        Create a filled region between upper and lower bounds.

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
        raise NotImplementedError

    @abstractmethod
    def heatmap(self, x, y, z, colorscale="viridis"):
        """
        Create a heatmap.

        Parameters
        ----------
        x, y : array-like
            Coordinate values.
        z : array-like
            Heatmap values.
        colorscale : str, optional
            Colour scale name.

        Returns
        -------
        object
            Backend-specific heatmap trace or figure.
        """
        raise NotImplementedError

    @abstractmethod
    def histogram_plot(self, x, name, style=None):
        """
        Create a histogram.

        Parameters
        ----------
        x : array-like
            Data to bin.
        name : str
            Histogram label.
        style : dict, optional
            Histogram styling options.
        """
        raise NotImplementedError

    @abstractmethod
    def line(self, x=None, y=None, label=None, style=None):
        """
        Create a line plot trace.

        Parameters
        ----------
        x, y : array-like, optional
            Coordinates of the line.
        label : str, optional
            Trace label.
        style : dict, optional
            Line styling options.

        Returns
        -------
        object
            Backend-specific line trace.
        """
        raise NotImplementedError

    @abstractmethod
    def scatter(self, x, y, colors, labels=None, colorscale="Greys"):
        """
        Create a scatter plot.

        Parameters
        ----------
        x, y : array-like
            Point coordinates.
        colors : array-like
            Values or colours associated with each point.
        labels : array-like, optional
            Point labels.
        colorscale : str, optional
            Colour scale name.
        """
        raise NotImplementedError

    @abstractmethod
    def show_table(self, header, values, title):
        """
        Display tabular data.

        Parameters
        ----------
        header : list
            Column headers.
        values : list
            Table contents.
        title : str
            Table title.
        """
        raise NotImplementedError

    @abstractmethod
    def vline(self, fig, x, style=None):
        """
        Add a vertical reference line to a figure.

        Parameters
        ----------
        fig : object
            Target figure.
        x : float
            X-coordinate of the line.
        style : dict, optional
            Line styling options.
        """
        raise NotImplementedError
