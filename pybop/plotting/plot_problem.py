import sys

import numpy as np

import pybop


def quick_plot(problem, parameter_values=None, show=True, **layout_kwargs):
    """
    Quickly plot the target dataset against optimised model output.

    Generates an interactive plot comparing the simulated model output with
    an optional target dataset and visualises uncertainty.

    Parameters
    ----------
    problem : object
        Problem object with dataset and signal attributes.
    parameter_values : array-like
        Optimised (or example) parameter values.
    show : bool, optional
        If True, the figure is shown upon creation (default: True).
    **layout_kwargs : optional
            Valid Plotly layout keys and their values,
            e.g. `xaxis_title="Time / s"` or
            `xaxis={"title": "Time / s", "titlefont_size": 18}`.

    Returns
    -------
    plotly.graph_objs.Figure
        The Plotly figure object for the scatter plot.
    """
    if parameter_values is None:
        parameter_values = problem.x0

    # Extract the time data and evaluate the model's output and target values
    xaxis_data = problem.time_data()
    model_output = problem.evaluate(parameter_values)
    target_output = problem.target()

    # Create a plot for each output
    figure_list = []
    for i in problem.signal:
        default_layout_options = dict(
            title="Scatter Plot",
            xaxis_title="Time / s",
            yaxis_title=pybop.StandardPlot.remove_brackets(i),
        )

        # Create a plotting dictionary
        if isinstance(problem, pybop.DesignProblem):
            trace_name = "Optimised"
            opt_time_data = model_output["Time [s]"]
        else:
            trace_name = "Model"
            opt_time_data = xaxis_data

        plot_dict = pybop.StandardPlot(
            x=opt_time_data,
            y=model_output[i],
            layout_options=default_layout_options,
            trace_names=trace_name,
        )

        target_trace = plot_dict.create_trace(
            x=xaxis_data,
            y=target_output[i],
            name="Reference",
            mode="markers",
            showlegend=True,
        )
        plot_dict.traces.append(target_trace)

        if isinstance(problem, pybop.FittingProblem):
            # Compute the standard deviation as proxy for uncertainty
            plot_dict.sigma = np.std(model_output[i] - target_output[i])

            # Convert x and upper and lower limits into lists to create a filled trace
            x = xaxis_data.tolist()
            y_upper = (model_output[i] + plot_dict.sigma).tolist()
            y_lower = (model_output[i] - plot_dict.sigma).tolist()

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
        if "ipykernel" in sys.modules and show:
            fig.show("svg")
        elif show:
            fig.show()

        figure_list.append(fig)

    return figure_list
