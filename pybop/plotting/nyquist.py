import sys

from pybop import StandardPlot
from pybop.parameters.parameter import Inputs


def nyquist(problem, problem_inputs: Inputs = None, show=True, **layout_kwargs):
    """
    Generates Nyquist plots for the given problem by evaluating the model's output and target values.

    Parameters
    ----------
    problem : pybop.BaseProblem
        An instance of a problem class (e.g., `pybop.EISProblem`) that contains the parameters and methods
        for evaluation and target retrieval.
    problem_inputs : Inputs, optional
        Input parameters for the problem. If not provided, the default parameters from the problem
        instance will be used. These parameters are verified before use (default is None).
    show : bool, optional
        If True, the plots will be displayed. If running in an IPython kernel (e.g., Jupyter Notebook),
        the plots will be shown using SVG format for better quality (default is True).
    **layout_kwargs : dict, optional
        Additional keyword arguments for customising the plot layout. These arguments are passed to
        `fig.update_layout()`.

    Returns
    -------
    list
        A list of plotly `Figure` objects, each representing a Nyquist plot for the model's output and target values.

    Notes
    -----
    - The function extracts the real part of the impedance from the model's output and the real and imaginary parts
      of the impedance from the target output.
    - For each signal in the problem, a Nyquist plot is created with the model's impedance plotted as a scatter plot.
    - An additional trace for the reference (target output) is added to the plot.
    - The plot layout can be customised using `layout_kwargs`.

    Example
    -------
    >>> problem = pybop.EISProblem()
    >>> nyquist_figures = nyquist(problem, show=True, title="Nyquist Plot", xaxis_title="Real(Z)", yaxis_title="Imag(Z)")
    >>> # The plots will be displayed and nyquist_figures will contain the list of figure objects.
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
