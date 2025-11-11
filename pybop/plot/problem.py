import numpy as np

from pybop.costs.design_cost import DesignCost
from pybop.costs.error_measures import ErrorMeasure
from pybop.parameters.parameter import Inputs
from pybop.plot.standard_plots import StandardPlot
from pybop.problems.meta_problem import MetaProblem
from pybop.problems.problem import Problem
from pybop.simulators.solution import Solution


def problem(
    problem: Problem,
    problem_inputs: Inputs = None,
    show: bool = True,
    **layout_kwargs,
):
    """
    Produce a quick plot of the target dataset against optimised model output.

    Generates an interactive plot comparing the simulated model output with
    an optional target dataset and visualises uncertainty.

    Parameters
    ----------
    problem : pybop.Problem
        Problem object with dataset and targets attributes.
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
    if not isinstance(problem_inputs, dict):
        problem_inputs = problem.parameters.to_dict(problem_inputs)

    domain = problem.domain
    if problem.domain_data is None:
        # Simulate the model for the both the initial and the given inputs
        target = problem.target
        problem.target = target + [domain]
        initial_inputs = problem.simulator.parameters.to_dict("initial")
        target_output = problem.simulate(initial_inputs)
        target_domain = target_output[domain].data
        model_output = problem.simulate(problem_inputs)
        model_domain = model_output[domain].data
        problem.target = target
    else:
        # Extract the time data and simulate the model for the given inputs
        target_output = Solution()
        for target in problem.target:
            target_output.set_solution_variable(
                target, data=problem.target_data[target]
            )
        target_domain = problem.domain_data
        model_output = problem.simulate(problem_inputs)
        model_domain = target_domain[: len(model_output[target].data)]

    # Create a plot for each output
    figure_list = []
    for var in problem.target:
        # Create a plot dictionary
        plot_dict = StandardPlot(
            layout_options=dict(
                title="Scatter Plot",
                xaxis_title="Time / s",
                yaxis_title=StandardPlot.remove_brackets(var),
            )
        )

        model_trace = plot_dict.create_trace(
            x=model_domain,
            y=model_output[var].data,
            name="Optimised" if isinstance(problem.cost, DesignCost) else "Model",
            mode="markers" if isinstance(problem, MetaProblem) else "lines",
            showlegend=True,
        )
        plot_dict.traces.append(model_trace)

        target_trace = plot_dict.create_trace(
            x=target_domain,
            y=target_output[var].data,
            name="Reference",
            mode="markers",
            showlegend=True,
        )
        plot_dict.traces.append(target_trace)

        if isinstance(problem.cost, ErrorMeasure) and len(
            model_output[var].data
        ) == len(target_output[var].data):
            # Compute the standard deviation as proxy for uncertainty
            plot_dict.sigma = np.std(model_output[var].data - target_output[var].data)

            # Convert x and upper and lower limits into lists to create a filled trace
            x = target_domain.tolist()
            y_upper = (model_output[var].data + plot_dict.sigma).tolist()
            y_lower = (model_output[var].data - plot_dict.sigma).tolist()

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
