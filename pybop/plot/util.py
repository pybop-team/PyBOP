import textwrap
import warnings

import numpy as np

import pybop.plot


def use_backend(backend):
    """
    Select a plotting backend to be used for all subsequent plots.

    Parameters
    ----------
    backend : str or pybop.plot.backends.PlotBackend
        The plotting backend to be used.
    """
    err_msg = (
        f"Plotting backend {backend} is not available. The current backend has not been updated. \n"
        f"The current backend is set to {pybop.plot.current_backend}"
    )

    if backend.lower() in ["matplotlib", "plotly"] or isinstance(
        backend, pybop.plot.backends.PlotBackend
    ):
        pybop.plot.current_backend = backend

    else:
        raise ModuleNotFoundError(err_msg)


def get_backend(backend=None):
    """
    Get instance of PlotBackend class for a given plotting backend

    Parameters
    ----------
    backend : str or pybop.plot.backends.PlotBackend, optional
        The plotting backend to be used (default: pybop.plot.current_backend)
    """
    if backend is None:
        backend = pybop.plot.current_backend
    elif isinstance(backend, pybop.plot.backends.PlotBackend):
        return backend

    err_msg = f"Plotting backend {backend} is not available."
    if backend.lower() == "matplotlib":
        return pybop.plot.backends.MatplotlibBackend()
    elif backend.lower() == "plotly":
        return pybop.plot.backends.PlotlyBackend()
    else:
        raise ModuleNotFoundError(err_msg)


def get_backend_from_figure(backend=None, figures=None):
    """
    Get instance of PlotBackend class from a provided figure or from a specified backend.
    If both are provided, the figure's backend takes precedence.

    Parameters
    ----------
    backend : str or pybop.plot.backends.PlotBackend, optional
        The plotting backend to be used (default: pybop.plot.current_backend)
    figures: figure object, optional
        Figure for plotting. If not provided a new figure is created

    Returns
    -------
    pybop.plot.backends.PlotBackend
        Instance of the selected plotting backend.
    """

    if figures is not None and len(np.atleast_1d(figures)) > 0:
        # Determine the backend from the provided figure
        if hasattr(figures, "__len__"):
            figures = figures[0]
        if "matplotlib" in str(type(figures)).lower():
            figure_backend = "matplotlib"
        elif "plotly" in str(type(figures)).lower():
            figure_backend = "plotly"
        else:
            raise ValueError(
                f"Could not determine the backend from the provided figure of type {type(figures)}"
            )
        # If a backend is provided, check if it matches the figure's backend
        if backend is not None:
            backend_str = (
                backend.name
                if isinstance(backend, pybop.plot.backends.PlotBackend)
                else backend.lower()
            )

            if backend_str != figure_backend:
                warnings.warn(
                    f"Backend {backend} does not match the provided figure's backend {figure_backend}. "
                    "Using the figure's backend.",
                    UserWarning,
                    stacklevel=2,
                )
                # Use the figure's backend if they don't match
                backend = figure_backend
        else:
            # If no backend is provided, use the figure's backend
            backend = figure_backend

    return get_backend(backend)


def parse_data(x, y):
    """
    Check the type and dimensions of the data and convert if necessary to a list
    of 'things plotly can take', e.g. numpy arrays or lists of numbers.

    Parameters
    ----------
    x : list or np.ndarray, optional
        X-axis data points.
    y : list or np.ndarray, optional
        Primary Y-axis data points for simulated model output.
    """
    if isinstance(x, list):
        # If it's a list of numpy arrays, it's fine
        # If it's a list of lists, it's fine
        # If it's neither, it's a list of numbers that we need to wrap
        if not isinstance(x[0], np.ndarray) and not isinstance(x[0], list):
            x = [x]
    elif isinstance(x, np.ndarray):
        x = np.squeeze(x)
        if x.ndim == 1:
            x = [x]
        else:
            x = x.tolist()
    if isinstance(y, list):
        if not isinstance(y[0], np.ndarray) and not isinstance(y[0], list):
            y = [y]
    if isinstance(y, np.ndarray):
        y = np.squeeze(y)
        if y.ndim == 1:
            y = [y]
        else:
            y = y.tolist()
    if len(x) > 1 and len(x) != len(y):
        raise ValueError(
            "Input x should have either one data series or the same number as y."
        )
    return x, y


def remove_brackets(s):
    """
    Remove square brackets from a string and replace with forward slashes
    as per section 7.1 of the SI Handbook
    """
    # If s is an iterable (but not a string), apply the function recursively to each element
    if hasattr(s, "__iter__") and not isinstance(s, str):
        return type(s)(remove_brackets(i) for i in s)
    elif isinstance(s, str):
        start = s.find("[")
        end = s.find("]")
        if start != -1 and end != -1:
            char_in_brackets = s[start + 1 : end]
            return s[:start] + " / " + char_in_brackets + s[end + 1 :]
    return s


def wrap_text(text, width, backend="matplotlib"):
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
    if backend == "plotly":
        return wrapped_text.replace("\n", "<br>")
    else:
        return wrapped_text
