import sys

from pybop import DesignProblem, StandardPlot
from pybop.parameters.parameter import Inputs


def nyquist(problem, problem_inputs: Inputs = None, show=True, **layout_kwargs):
    """
    TODO: docstring
    """
    if problem_inputs is None:
        problem_inputs = problem.parameters.as_dict()
    else:
        problem_inputs = problem.parameters.verify(problem_inputs)

    # Extract the time data and evaluate the model's output and target values
    model_output = problem.evaluate(problem_inputs)
    domain_data = model_output["Impedance"].real
    target_output = problem.get_target()

    # Create a plot for each output
    figure_list = []
    for i in problem.signal:
        default_layout_options = dict(
            title="Scatter Plot",
            xaxis_title="Z_re / Ohm",
            yaxis_title="Z_i / Ohm",
        )

        # Create a plotting dictionary
        if isinstance(problem, DesignProblem):
            trace_name = "Optimised"
        else:
            trace_name = "Model"

        plot_dict = StandardPlot(
            x=domain_data,
            y=-model_output[i].imag,
            layout_options=default_layout_options,
            trace_names=trace_name,
        )

        target_trace = plot_dict.create_trace(
            x=target_output[i].real,
            y=-target_output[i].imag,
            name="Reference",
            mode="markers",
            showlegend=True,
        )
        plot_dict.traces.append(target_trace)

        # Generate the figure and update the layout
        fig = plot_dict(show=False)
        fig.update_layout(**layout_kwargs)
        if "ipykernel" in sys.modules and show:
            fig.show("svg")
        elif show:
            fig.show()

        figure_list.append(fig)

    return figure_list
