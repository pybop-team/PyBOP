import numpy as np

from pybop import OptimisationResult


def classify_using_hessian(
    result: OptimisationResult,
    dx=None,
    cost_tolerance: float = 1e-5,
    normalise: bool = True,
):
    """
    A simple check for parameter correlations based on numerical approximation
    of the Hessian matrix at the optimal point using central finite differences.

    Parameters
    ----------
    result : OptimisationResult
        The optimisation result.
    dx : array-like, optional
        An array of small positive values used to check proximity to the parameter
        bounds and as the perturbation distance in the finite difference calculations.
    cost_tolerance : float, optional
        A small positive tolerance used for cost value comparisons (default: 1e-5).
    normalise : bool, optional
        If True, the Hessian is scaled by the step size in the parameters so that the
        Hessian entries are in the same unit as the cost values (default: True).
    """
    x = result.x
    dx = np.asarray(dx) if dx is not None else np.maximum(x, 1e-40) * 1e-2
    best_cost = result.best_cost
    problem = result.optim.problem
    parameters = problem.parameters
    minimising = result.minimising
    cost_tolerance = float(cost_tolerance)

    # Prepare outputs
    stationarity_confirmed = False
    cfd_hessian = np.full((2, 2), np.nan, dtype=float)
    eigenvalues = np.array([np.nan, np.nan], dtype=float)
    eigenvectors = np.full((2, 2), np.nan, dtype=float)

    def cost(x):
        return problem.evaluate(x).values

    n = len(x)
    if n != 2 or len(dx) != n:
        raise ValueError(
            "The function classify_using_hessian currently only works in the case "
            "of 2 parameters, and dx must have the same length as x."
        )

    # Get a list of parameter names for use in the output message
    names = parameters.names

    # Evaluate the cost for a grid of surrounding points
    costs = np.zeros((3, 3, 2))
    for i in np.arange(0, 3):
        for j in np.arange(0, 3):
            if i == j == 1:
                costs[1, 1, 0] = best_cost
                costs[1, 1, 1] = best_cost
            else:
                costs[i, j, 0] = cost(x + np.multiply([i - 1, j - 1], dx))[0]
                costs[i, j, 1] = cost(x + np.multiply([i - 1, j - 1], 2 * dx))[0]

    def check_proximity_to_bounds(parameters, x, dx, names) -> str:
        bounds = parameters.get_bounds()
        if bounds is not None:
            for i, value in enumerate(x):
                if value > bounds["upper"][i] - dx[i]:
                    return f" The result is near the upper bound of {names[i]}."

                if value < bounds["lower"][i] + dx[i]:
                    return f" The result is near the lower bound of {names[i]}."
        return ""

    # Classify the result
    if (minimising and np.any(costs < best_cost)) or (
        not minimising and np.any(costs > best_cost)
    ):
        message = "The optimiser has not converged to a stationary point."
        message += check_proximity_to_bounds(parameters, x, dx, names)

    elif not np.all([np.isfinite(cost) for cost in costs]):
        message = "Classification cannot proceed due to infinite cost value(s)."
        message += check_proximity_to_bounds(parameters, x, dx, names)

    else:
        # By default, normalise Hessian with respect to the finite differencing step in the
        # parameter values, in order to compute eigenvalues in the same unit as the cost

        # Estimate the normalised Hessian using second-order accurate central finite differences
        # cfd_hessian[0, 0] = costs[2,1,0] - 2 * costs[1,1,0] + costs[0,1,0]
        # cfd_hessian[0, 1] = (costs[2,2,0] - costs[2,0,0] + costs[0,0,0] - costs[0,2,0]) / 4
        # cfd_hessian[1, 0] = cfd_hessian[0, 1]
        # cfd_hessian[1, 1] = costs[1,2,0] - 2 * costs[1,1,0] + costs[1,0,0]

        # Estimate the normalised Hessian using fourth-order accurate central finite differences
        cfd_hessian[0, 0] = (
            -costs[2, 1, 1]
            + 16 * costs[2, 1, 0]
            - 30 * costs[1, 1, 0]
            + 16 * costs[0, 1, 0]
            - costs[0, 1, 1]
        ) / 12
        cfd_hessian[0, 1] = (
            -(costs[2, 2, 1] - costs[2, 0, 1] + costs[0, 0, 1] - costs[0, 2, 1])
            + 16 * (costs[2, 2, 0] - costs[2, 0, 0] + costs[0, 0, 0] - costs[0, 2, 0])
        ) / 48
        cfd_hessian[1, 0] = cfd_hessian[0, 1]
        cfd_hessian[1, 1] = (
            -costs[1, 2, 1]
            + 16 * costs[1, 2, 0]
            - 30 * costs[1, 1, 0]
            + 16 * costs[1, 0, 0]
            - costs[1, 0, 1]
        ) / 12

        if not normalise:
            # Replace the normalised Hessian by the true Hessian
            cfd_hessian[0, 0] /= dx[0] ** 2
            cfd_hessian[0, 1] /= dx[0] * dx[1]
            cfd_hessian[1, 0] /= dx[0] * dx[1]
            cfd_hessian[1, 1] /= dx[1] ** 2

        # Compute the eigenvalues and sort into ascending order
        eigenvalues, eigenvectors = np.linalg.eig(cfd_hessian)
        idx = eigenvalues.argsort()
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Classify the result
        if np.all(eigenvalues > cost_tolerance):
            stationarity_confirmed = True
            message = "The optimiser has located a minimum."
        elif np.all(eigenvalues < -cost_tolerance):
            stationarity_confirmed = True
            message = "The optimiser has located a maximum."
        elif np.all(np.abs(eigenvalues) > cost_tolerance):
            stationarity_confirmed = True
            message = "The optimiser has located a saddle point."
        elif np.all(np.abs(eigenvalues) < cost_tolerance):
            message = f"The cost variation is smaller than the cost tolerance: {cost_tolerance}."
        else:
            # One eigenvalue is too small to classify with certainty
            message = "The cost variation is too small to classify with certainty."

        # Check for parameter correlations
        if np.any(np.abs(eigenvalues) > cost_tolerance):
            if np.allclose(eigenvectors[0], np.array([1, 0])):
                message += f" The cost is insensitive to a change of {dx[0]:.2g} in {names[0]}."
            elif np.allclose(eigenvectors[0], np.array([0, 1])):
                message += f" The cost is insensitive to a change of {dx[1]:.2g} in {names[1]}."
            else:
                diagonal_costs = [
                    cost(x - np.multiply(eigenvectors[:, 0], dx)),
                    cost(x + np.multiply(eigenvectors[:, 0], dx)),
                ]
                if np.allclose(best_cost, diagonal_costs, atol=cost_tolerance, rtol=0):
                    message += " There may be a correlation between these parameters."

        if normalise:
            # Now, after the checks, replace the normalised Hessian by the true Hessian
            cfd_hessian[0, 0] /= dx[0] ** 2
            cfd_hessian[0, 1] /= dx[0] * dx[1]
            cfd_hessian[1, 0] /= dx[0] * dx[1]
            cfd_hessian[1, 1] /= dx[1] ** 2

            # Scale the results to match the true Hessian
            for k in range(eigenvectors.shape[1]):
                vec = eigenvectors[:, k] * dx
                eigenvectors[:, k] = vec / np.linalg.norm(vec)
            eigenvalues /= dx**2

    print(message)

    # Pack everything useful into a dictionary
    return {
        "message": message,
        "cost": cost,
        "cfd_hessian": cfd_hessian,
        "stationarity_confirmed": stationarity_confirmed,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "x": x,
        "dx": dx,
        "names": names,
        "best_cost": best_cost,
    }


def plot_hessian_eigenvectors(info, steps: int = 10):
    """
    A function to plot the eigenvectors computed for the Hessian at an optimal point.

    Parameters
    ----------
    info : dict
        The output from pybop.classify_using_Hessian.
    steps : int
        Grid resolution per axis.
    """
    import matplotlib.pyplot as plt

    cost = info["cost"]
    x = info["x"]
    dx = info["dx"]
    names = info["names"]
    eigenvalues = info["eigenvalues"]
    eigenvectors = info["eigenvectors"]

    # Build a plotting span around x
    span_multiplier = 4.0
    span0 = (x[0] - span_multiplier * dx[0], x[0] + span_multiplier * dx[0])
    span1 = (x[1] - span_multiplier * dx[1], x[1] + span_multiplier * dx[1])
    param0 = np.linspace(span0[0], span0[1], steps)
    param1 = np.linspace(span1[0], span1[1], steps)

    # Evaluate cost on the grid
    Z = np.empty((steps, steps), dtype=float)
    for i in range(steps):
        for j in range(steps):
            p = np.array([param0[i], param1[j]], dtype=float)
            try:
                Z[i, j] = float(cost(p))
            except Exception:
                Z[i, j] = np.nan

    # Pack everything useful into a dictionary
    info.update(
        {
            "span0": span0,
            "span1": span1,
            "param0": param0,
            "param1": param1,
            "Z": Z,
        }
    )

    # Cost contours
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(
        [x[0]], [x[1]], marker="x", s=60, label=f"Result (cost={info['best_cost']:.3g})"
    )
    Z[~np.isfinite(Z)] = np.nan
    if Z.size == 0:
        ax.text(
            0.5, 0.5, "No finite cost values on contour grid", ha="center", va="center"
        )
    else:
        vmin, vmax = np.nanmin(Z), np.nanmax(Z)
        levels = np.linspace(vmin, vmax, 10)
        cs = ax.contour(param0, param1, Z.T, levels=levels)
        ax.clabel(cs, inline=1, fontsize=8)

    # Add eigenvectors
    if info["stationarity_confirmed"]:
        colours = ["red", "purple"]
        for k, val in enumerate(eigenvalues):
            vec = eigenvectors[:, k]
            if np.isfinite(vec).all():
                ax.axline(
                    x,
                    slope=vec[1] / vec[0],
                    color=colours[k],
                    linestyle="--",
                    linewidth=1.2,
                    label=f"eig {k} (λ={val:.3g})",
                )

    ax.set_xlim(info["span0"])
    ax.set_ylim(info["span1"])
    ax.set_xlabel(names[0])
    ax.set_ylabel(names[1])
    ax.set_title("Cost contours")
    ax.legend(loc="best")

    return fig, ax
