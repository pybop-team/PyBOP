import math

import numpy as np

from pybop.costs.design_cost import DesignCost
from pybop.costs.error_measures import ErrorMeasure
from pybop.parameters.parameter import Inputs
from pybop.plot.util import get_backend_from_figure, remove_brackets
from pybop.problems.meta_problem import MetaProblem
from pybop.problems.problem import Problem
from pybop.processing.dataset import parse_impedance_variables
from pybop.simulators.solution import Solution


def problem(
    problem: Problem,
    inputs: Inputs = None,
    title="Scatter Plot",
    show: bool = True,
    backend: str = None,
    figures=None,
    axes=None,
):
    """
    Produce a quick plot of the target dataset against optimised model output.

    Generates an interactive plot comparing the simulated model output with
    an optional target dataset and visualises uncertainty.

    Parameters
    ----------
    problem : pybop.Problem
        Problem object with dataset and targets attributes.
    inputs : Inputs
        Optimised (or example) parameter values.
    title: str, optional:
        The title of the plot (default: "Scatter Plot")
    show : bool, optional
        If True, the figure is shown upon creation (default: True).
    backend: str or pybop.plot.backends.PlotBackend, optional
        The plotting backend to be used.
    figures: figure object, optional
        Figure for plotting. If not provided a new figure is created for each problem.
        Can be a single figure or one figure per target.
    axes: axis, optional
        The axes to be used for plotting. One axis per target is expected.
        plotly: axes expected to be of the form list of tuple(row, col)

    Returns
    -------
    None: if show is True
    Figure or list of figures: if show is False
        A single figure or a list of figures containing the plots for each target in the
        problem. If show is True, the figures will be displayed and None will be returned.
    """
    if inputs is None:
        inputs = problem.parameters.to_dict()
    elif not isinstance(inputs, dict):
        raise TypeError(f"Expecting a dictionary, received {type(inputs)}")

    domain = problem.domain
    if problem.domain_data is None:
        # Simulate the model for both the initial and the given inputs
        target = problem.target
        problem.set_target(target + [domain])
        initial_inputs = problem.parameters.to_dict("initial")
        target_output = problem.simulate(initial_inputs)
        target_domain = target_output[domain].data
        model_output = problem.simulate(inputs)
        model_domain = model_output[domain].data
        problem.set_target(target)
    else:
        # Extract the time data and simulate the model for the given inputs
        target_output = Solution()
        for target in problem.target:
            target_output.set_solution_variable(
                target, data=problem.target_data[target]
            )
        target_domain = problem.domain_data
        model_output = problem.simulate(inputs)
        model_domain = target_domain[: len(model_output[target].data)]

    # Import plotting backend
    backend = get_backend_from_figure(backend, figures)

    # Impedance variables are plotted as a Nyquist plot per acquisition, not as a series
    # over the domain, so separate them from the remaining targets
    frequencies, real_variables, imaginary_variables = parse_impedance_variables(
        problem.target
    )
    targets = [
        var
        for var in problem.target
        if var not in set(real_variables) | set(imaginary_variables)
    ]
    acquisitions = (
        _acquisition_indices(target_output, real_variables, imaginary_variables)
        if len(frequencies) > 0
        else []
    )

    # Process input
    figures, axes, create_figure, _ = backend.parse_input_axes(
        figures,
        axes,
        num_plots=len(targets),
        allow_single_axis=False,
    )

    # Create a plot for each output
    for i, var in enumerate(targets):
        ax = axes[i % len(axes)]
        if create_figure:
            fig = backend.create_figure(
                style={"bg_color": "white", "width": 600, "height": 600},
            )
            figures = np.append(figures, fig)
        else:
            fig = figures[i % len(figures)]

        backend.update_axes_titles(
            fig, ax, remove_brackets(domain), remove_brackets(var)
        )
        backend.update_plot_titles(fig, ax, title)
        traces = []

        model_trace = backend.line(
            x=model_domain,
            y=model_output[var].data,
            label="Optimised" if isinstance(problem.cost, DesignCost) else "Model",
            style={
                "linestyle": "none" if isinstance(problem, MetaProblem) else "solid",
                "marker": "." if isinstance(problem, MetaProblem) else "none",
            },
        )
        traces.append(model_trace)

        target_trace = backend.line(
            x=target_domain,
            y=target_output[var].data,
            label="Reference",
            style={"linestyle": "none", "marker": "."},
        )
        traces.append(target_trace)

        if isinstance(problem.cost, ErrorMeasure) and len(
            model_output[var].data
        ) == len(target_output[var].data):
            # Compute the standard deviation as proxy for uncertainty
            sigma = np.std(model_output[var].data - target_output[var].data)

            # Convert x and upper and lower limits into lists to create a filled trace
            x = target_domain.tolist()
            y_upper = (model_output[var].data + sigma).tolist()
            y_lower = (model_output[var].data - sigma).tolist()

            fill_trace = backend.fill_between(x, y_upper, y_lower, color="#FFE5CC")
            traces.append(fill_trace)

        # Reverse the order of the traces to put the model on top
        traces = traces[::-1]

        for trace in traces:
            backend.plot_trace(trace, fig, ax=ax)

        backend.legend(fig, axes=ax)

        # Generate the figure and update the layout
        if show:
            backend.show_figure(fig)

    # Collect the acquired spectra into a single figure of Nyquist subplots
    if len(acquisitions) > 0:
        num_cols = int(math.ceil(math.sqrt(len(acquisitions))))
        num_rows = int(math.ceil(len(acquisitions) / num_cols))
        impedance_figure, impedance_axes = backend.make_subplots(
            num_rows=num_rows,
            num_cols=num_cols,
            num_plots=len(acquisitions),
            title=title,
            style={
                "bg_color": "white",
                "width": 400 * num_cols,
                "height": 400 * num_rows,
            },
        )
        backend.update_axes_titles(
            impedance_figure,
            impedance_axes,
            r"$Z_{re} / \Omega$",
            r"$-Z_{im} / \Omega$",
        )
        backend.update_plot_titles(
            impedance_figure,
            impedance_axes,
            [
                f"{remove_brackets(domain)} = {target_domain[row]:g}"
                for row in acquisitions
            ],
        )

        # Fixed styles, matching plot.nyquist, so that the colours mean the same thing
        # in every subplot rather than following the shared colour cycle
        impedance_styles = (
            (
                target_output,
                "Reference",
                {
                    "linestyle": "none",
                    "marker": "o",
                    "fillstyle": "none",
                    "markeredgecolor": "#636EFA",
                    "color": "#636EFA",
                },
            ),
            (model_output, "Model", {"linestyle": "solid", "color": "#00CC96"}),
        )

        for ax, row in zip(impedance_axes, acquisitions, strict=True):
            for output, label, style in impedance_styles:
                backend.plot_trace(
                    backend.line(
                        x=[output[name].data[row] for name in real_variables],
                        y=[-output[name].data[row] for name in imaginary_variables],
                        label=label,
                        style=style,
                    ),
                    impedance_figure,
                    ax=ax,
                )

            # A Nyquist plot is only readable with an equal aspect ratio
            backend.equal_aspect(impedance_figure, ax=ax)
            backend.legend(impedance_figure, axes=ax)

        figures = np.append(figures, impedance_figure)
        if show:
            backend.show_figure(impedance_figure)

    if not show:
        return figures[0] if len(figures) == 1 else figures


def _acquisition_indices(
    target_output, real_variables: list[str], imaginary_variables: list[str]
) -> np.ndarray:
    """Return the indices of the domain points at which a spectrum was acquired."""
    measured = np.asarray(
        [target_output[name].data for name in (*real_variables, *imaginary_variables)]
    )
    return np.flatnonzero(np.any(measured != 0.0, axis=0))
