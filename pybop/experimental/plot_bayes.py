from multiprocessing import Pool

import numpy as np
from base_bayes_optimiser import BayesianOptimisationResult
from ep_bolfi.utility.preprocessing import combine_parameters_to_try

from pybop import DesignProblem
from pybop.plot.standard_plots import StandardPlot


def bayes(problem, results: BayesianOptimisationResult, show=True, **layout_kwargs):
    """
    Plot the target dataset against optimised model distribution.

    Generates an interactive plot comparing the predictive posterior
    with an optinal target dataset. If `results` has lower and upper
    bounds, the range of most extreme model realizations will be
    visualized. If `results` has a posterior distribution, randomly
    drawn variates of it will visualize the predictive posterior.

    Parameters
    ----------
    problem: object
        Problem object with dataset and signal attributes.
    results : BayesianOptimisationResult
        Optimised parameter distribution.
    show : bool, optional
        If True, the figure is shown upon creation (default: True).
    **layout_kwargs : optional
        Valid Plotly layout keys and their values,
        e.g. `xaxis_title="Time / s"` or
        `xaxis={"title": "Time / s", "titlefont_size": 18}`.

    Returns
    -------
    plotly.graph_objs.Figure
        The Plotly figure object for the predictive posterior plot.
    """

    # Extract the time data and evaluate the model's output and target values
    xaxis_data = problem.domain_data
    model_output = problem.evaluate(results.x)
    target_output = problem.get_target()

    # Explore the range of model realizations
    if results.lower_bounds is not None and results.upper_bounds is not None:
        lower_bounds_dict = {k: v for k, v in enumerate(results.lower_bounds)}
        x_dict = {k: v for k, v in enumerate(results.x)}
        upper_bounds_dict = {k: v for k, v in enumerate(results.upper_bounds)}
        bounds_dict = {
            k: (lower_bounds_dict[k], upper_bounds_dict[k]) for k in range(len(x_dict))
        }

        permutations = combine_parameters_to_try(x_dict, bounds_dict)[0]
        for i in range(len(permutations)):
            permutations[i] = np.asarray(
                [permutations[i][k] for k in range(len(permutations[i]))]
            )

        with Pool() as p:
            extremes = p.map(problem.evaluate, permutations)
            extremes = {
                i: [extremes[j][i] for j in range(len(extremes))]
                for i in problem.signal
            }

    # Explore the predictive posterior
    if results.posterior is not None:
        resamples = results.posterior.rvs(
            2 ** (2 + len(results.x)), apply_transform=True
        )
        if resamples.ndim == 1:
            resamples = np.atleast_2d(resamples).T

        with Pool() as p:
            predictions = p.map(problem.evaluate, resamples)
            predictions = {
                i: [predictions[j][i] for j in range(len(predictions))]
                for i in problem.signal
            }

    # Create a plot for each output
    figure_list = []
    for i in problem.signal:
        default_layout_options = dict(
            title="Scatter Plot",
            xaxis_title="Time / s",
            yaxis_title=StandardPlot.remove_brackets(i),
        )

        # Create a plot dictionary
        if isinstance(problem, DesignProblem):
            trace_name = "Optimised"
            opt_domain_data = model_output["Time [s]"]
        else:
            trace_name = "Model"
            opt_domain_data = xaxis_data

        plot_dict = StandardPlot(
            x=opt_domain_data,
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

        if results.posterior is not None:
            x = xaxis_data.tolist()
            for realization in predictions[i]:
                target_trace = plot_dict.create_trace(
                    x=xaxis_data,
                    y=realization,
                    mode="lines",
                    showlegend=False,
                    line={"width": 0.2},
                )
                plot_dict.traces.append(target_trace)

        if results.lower_bounds is not None and results.upper_bounds is not None:
            x = xaxis_data.tolist()
            y_lower = np.min(extremes[i], axis=0).tolist()
            y_upper = np.max(extremes[i], axis=0).tolist()

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
