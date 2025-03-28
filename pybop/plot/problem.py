import jax.numpy as jnp
import numpy as np

from pybop import DesignProblem, FittingProblem, MultiFittingProblem
from pybop.parameters.parameter import Inputs
from pybop.plot.standard_plots import StandardPlot


def problem(problem, problem_inputs: Inputs = None, show=True, **layout_kwargs):
    """
    Produce a quick plot of the target dataset against optimised model output.

    Generates an interactive plot comparing the simulated model output with
    an optional target dataset and visualises uncertainty.

    Parameters
    ----------
    problem : object
        Problem object with dataset and signal attributes.
    problem_inputs : Inputs
        Optimised (or example) parameter values.
    show : bool, optional
        If True, the figure is shown upon creation (default: True).
    **layout_kwargs : optional
            Valid Plotly layout keys and their values,
            e.g. `xaxis_title="Time / s"` or
            `xaxis={"title": "Time [s]", font={"size":14}}`

    Returns
    -------
    plotly.graph_objs.Figure
        The Plotly figure object for the scatter plot.
    """
    if problem_inputs is None:
        problem_inputs = problem.parameters.as_dict()
    else:
        problem_inputs = problem.parameters.verify(problem_inputs)

    # Extract the time data and evaluate the model's output and target values
    domain = problem.domain
    domain_data = problem.domain_data
    model_output = problem.evaluate(problem_inputs)
    target_output = problem.get_target()

    # Convert model_output to np if Jax array
    if isinstance(model_output[problem.signal[0]], jnp.ndarray):
        model_output = {
            signal: np.asarray(model_output[signal]) for signal in problem.signal
        }

    # Create a plot for each output
    figure_list = []
    for signal in problem.signal:
        # Create a plot dictionary
        plot_dict = StandardPlot(
            layout_options=dict(
                title="Scatter Plot",
                xaxis_title="Time / s",
                yaxis_title=StandardPlot.remove_brackets(signal),
            )
        )

        model_trace = plot_dict.create_trace(
            x=model_output[domain]
            if domain in model_output.keys()
            else domain_data[: len(model_output[signal])],
            y=model_output[signal],
            name="Optimised" if isinstance(problem, DesignProblem) else "Model",
            mode="markers" if isinstance(problem, MultiFittingProblem) else "lines",
            showlegend=True,
        )
        plot_dict.traces.append(model_trace)

        target_trace = plot_dict.create_trace(
            x=domain_data,
            y=target_output[signal],
            name="Reference",
            mode="markers",
            showlegend=True,
        )
        plot_dict.traces.append(target_trace)

        if isinstance(problem, FittingProblem) and len(model_output[signal]) == len(
            target_output[signal]
        ):
            # Compute the standard deviation as proxy for uncertainty
            plot_dict.sigma = np.std(model_output[signal] - target_output[signal])

            # Convert x and upper and lower limits into lists to create a filled trace
            x = domain_data.tolist()
            y_upper = (model_output[signal] + plot_dict.sigma).tolist()
            y_lower = (model_output[signal] - plot_dict.sigma).tolist()

            fill_trace = plot_dict.create_trace(
                x=x + x[::-1],
                y=y_upper + y_lower[::-1],
                fill="toself",
                fillcolor="rgba(255,229,204,0.8)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=False,
            )
            plot_dict.traces.append(fill_trace)

        # Reverse the order of the traces to put the model on top
        plot_dict.traces = plot_dict.traces[::-1]

        # Generate the figure and update the layout
        fig = plot_dict(show=False)
        fig.update_layout(**layout_kwargs)
        if show:
            fig.show()

        figure_list.append(fig)

    return figure_list
