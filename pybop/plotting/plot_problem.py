import pybop
import numpy as np


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
            e.g. `xaxis_title="Time [s]"` or
            `xaxis={"title": "Time [s]", "titlefont_size": 18}`.

    Returns
    -------
    plotly.graph_objs.Figure
        The Plotly figure object for the scatter plot.
    """
    if parameter_values is None:
        parameter_values = problem.x0

    # Extract the time data and evaluate the model's output and target values
    time_data = problem.time_data()
    model_output = problem.evaluate(parameter_values)
    target_output = problem.target()

    # Ensure outputs have the same length
    len_diff = len(target_output) - len(model_output)
    if len_diff > 0:
        model_output = np.concatenate(
            (model_output, np.full([len_diff, np.shape(model_output)[1]], np.nan)),
            axis=0,
        )
    elif len_diff < 0:
        target_output = np.concatenate(
            (target_output, np.full([-len_diff, np.shape(target_output)[1]], np.nan)),
            axis=0,
        )

    # Create a plot for each output
    figure_list = []
    for i in range(0, problem.n_outputs):
        default_layout_options = dict(
            title="Scatter Plot", xaxis_title="Time [s]", yaxis_title=problem.signal[i]
        )

        # Create a plotting dictionary
        if isinstance(problem, pybop.DesignProblem):
            trace_name = "Optimised"
        else:
            trace_name = "Model"
        plot_dict = pybop.StandardPlot(
            x=time_data,
            y=model_output[:, i],
            layout_options=default_layout_options,
            trace_names=trace_name,
        )

        # Add the data as markers
        if isinstance(problem, pybop.DesignProblem):
            name = "Initial"
        else:
            name = "Target"
        target_trace = plot_dict.create_trace(
            x=time_data,
            y=target_output[:, i],
            name=name,
            mode="markers",
            showlegend=True,
        )
        plot_dict.traces.append(target_trace)

        # Compute the standard deviation as proxy for uncertainty
        plot_dict.sigma = np.std(model_output[:, i] - target_output[:, i])

        # Convert x and upper and lower limits into lists to create a filled trace
        x = time_data.tolist()
        y_upper = (model_output[:, i] + plot_dict.sigma).tolist()
        y_lower = (model_output[:, i] - plot_dict.sigma).tolist()
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
