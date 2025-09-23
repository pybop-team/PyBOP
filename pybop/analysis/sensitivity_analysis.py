import numpy as np
from SALib.analyze import sobol
from SALib.sample.sobol import sample


def sensitivity_analysis(
    problem, n_samples: int = 256, calc_second_order: bool = False
) -> dict:
    """
    Computes the parameter sensitivities on the combined cost function using
    SOBOL analysis from the SALib module [1].

    Parameters
    ----------
    problem : pybop.BaseProblem
        The optimisation problem.
    n_samples : int, optional
        Number of samples for SOBOL sensitivity analysis,
        performs best as order of 2, i.e. 128, 256, etc.
    calc_second_order : bool, optional
        Whether to calculate second-order sensitivities.

    References
    ----------
    .. [1] Iwanaga, T., Usher, W., & Herman, J. (2022). Toward SALib 2.0:
            Advancing the accessibility and interpretability of global sensitivity
            analyses. Socio-Environmental Systems Modelling, 4, 18155.
            doi:10.18174/sesmo.18155

    Returns
    -------
    Sensitivities : dict
    """

    salib_dict = {
        "names": problem.parameters.names,
        "bounds": problem.parameters.get_bounds_array(),
        "num_vars": len(problem.parameters),
    }

    # Create samples, compute cost
    param_values = sample(salib_dict, n_samples)
    costs = np.asarray(problem(param_values))

    return sobol.analyze(salib_dict, costs, calc_second_order=calc_second_order)
