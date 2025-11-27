from pybop import PybammEISProblem
from pybop.parameters.parameter import Inputs
from pybop.plot.standard_plots import StandardPlot


def nyquist(
    problem: PybammEISProblem,
    problem_inputs: Inputs | None = None,
    show: bool = True,
    **layout_kwargs,
):
    """
    Generates Nyquist plots for the given problem by evaluating the model's output and target values.

    Parameters
    ----------
    problem : pybop.Problem
        An instance of a problem class (e.g., `pybop.EISProblem`) that contains the parameters and methods
        for evaluation and target retrieval.
    problem_inputs : Inputs, optional
        Input parameters for the problem. If not provided, the default parameters from the problem
        instance will be used. These parameters are verified before use (default is None).
    show : bool, optional
        If True, the plots will be displayed.
    **layout_kwargs : dict, optional
        Additional keyword arguments for customising the plot layout. These arguments are passed to
        `fig.update_layout()`.

    Returns
    -------
    Figure
        A plotly `Figure` object representing a Nyquist plot for the model's output and target values.

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
    if not isinstance(problem, PybammEISProblem):
        raise TypeError(
            "The problem must be an instance of PybammEISProblem, "
            "e.g. pybop.EISProblem or pybop.PybammEISProblem."
        )
    if problem_inputs is None:
        problem_inputs = problem.parameters.to_dict()
    else:
        problem.parameters.update(values=problem_inputs)
        problem_inputs = problem.parameters.to_dict()

    model_output = problem.simulate(problem_inputs)
    domain_data = model_output.real
    target_output = problem.fitting_data

    default_layout_options = dict(
        title="Nyquist Plot",
        font=dict(family="Arial", size=14),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(
            title=dict(text="Z<sub>re</sub> / Ω", font=dict(size=16), standoff=15),
            showline=True,
            linewidth=2,
            linecolor="black",
            mirror=True,
            ticks="outside",
            tickwidth=2,
            tickcolor="black",
            ticklen=5,
        ),
        yaxis=dict(
            title=dict(text="-Z<sub>im</sub> / Ω", font=dict(size=16), standoff=15),
            showline=True,
            linewidth=2,
            linecolor="black",
            mirror=True,
            ticks="outside",
            tickwidth=2,
            tickcolor="black",
            ticklen=5,
            scaleanchor="x",
            scaleratio=1,
        ),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor="rgba(255, 255, 255, 0.5)",
            bordercolor="black",
            borderwidth=1,
        ),
        width=600,
        height=600,
    )

    plot_dict = StandardPlot(
        x=domain_data,
        y=-model_output.imag,
        layout_options=default_layout_options,
        trace_names="Model",
    )

    plot_dict.traces[0].update(
        mode="lines+markers",
        line=dict(color="blue", width=2),
        marker=dict(size=8, color="blue", symbol="circle"),
    )

    if target_output is not None:
        target_trace = plot_dict.create_trace(
            x=target_output.real,
            y=-target_output.imag,
            name="Reference",
            mode="markers",
            marker=dict(size=8, color="red", symbol="circle-open"),
            showlegend=True,
        )
        plot_dict.traces.append(target_trace)

    fig = plot_dict(show=False)

    # Add minor gridlines
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="lightgray",
        minor=dict(showgrid=True, gridwidth=0.5, gridcolor="lightgray"),
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="lightgray",
        minor=dict(showgrid=True, gridwidth=0.5, gridcolor="lightgray"),
    )

    # Overwrite with user-kwargs
    fig.update_layout(**layout_kwargs)
    if show:
        fig.show()

    return fig
