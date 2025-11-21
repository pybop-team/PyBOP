import matplotlib.pyplot as plt
import numpy as np

from pybop import OptimisationResult


def classify_using_hessian(
    result: OptimisationResult,
    dx=None,
    cost_tolerance: float | None = 1e-5,
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
    """
    x = result.x
    dx = np.asarray(dx) if dx is not None else np.maximum(x, 1e-40) * 1e-2
    best_cost = result.best_cost
    problem = result.optim.problem
    parameters = problem.parameters
    minimising = result.minimising
    cost_tolerance = float(cost_tolerance) if cost_tolerance is not None else 0.0

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
                costs[i, j, 0] = cost(x + np.multiply([i - 1, j - 1], dx))
                costs[i, j, 1] = cost(x + np.multiply([i - 1, j - 1], 2 * dx))

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
        # Estimate the Hessian using second-order accurate central finite differences
        # cfd_hessian = np.zeros((2, 2))
        # cfd_hessian[0, 0] = costs[2,1,0] - 2 * costs[1,1,0] + costs[0,1,0]
        # cfd_hessian[0, 1] = (costs[2,2,0] - costs[2,0,0] + costs[0,0,0] - costs[0,2,0]) / 4
        # cfd_hessian[1, 0] = cfd_hessian[0, 1]
        # cfd_hessian[1, 1] = costs[1,2,0] - 2 * costs[1,1,0] + costs[1,0,0]

        # Estimate the Hessian using fourth-order accurate central finite differences
        cfd_hessian = np.zeros((2, 2))
        cfd_hessian[0, 0] = (
            (
                -costs[2, 1, 1]
                + 16 * costs[2, 1, 0]
                - 30 * costs[1, 1, 0]
                + 16 * costs[0, 1, 0]
                - costs[0, 1, 1]
            )
            / 12
            / (dx[0] ** 2)
        )
        cfd_hessian[0, 1] = (
            (
                -(costs[2, 2, 1] - costs[2, 0, 1] + costs[0, 0, 1] - costs[0, 2, 1])
                + 16
                * (costs[2, 2, 0] - costs[2, 0, 0] + costs[0, 0, 0] - costs[0, 2, 0])
            )
            / 48
            / (dx[0] * dx[1])
        )
        cfd_hessian[1, 0] = cfd_hessian[0, 1]
        cfd_hessian[1, 1] = (
            (
                -costs[1, 2, 1]
                + 16 * costs[1, 2, 0]
                - 30 * costs[1, 1, 0]
                + 16 * costs[1, 0, 0]
                - costs[1, 0, 1]
            )
            / 12
            / (dx[1] ** 2)
        )

        # Compute the eigenvalues and sort into ascending order
        eigenvalues, eigenvectors = np.linalg.eig(cfd_hessian)
        idx = eigenvalues.argsort()
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Classify the result
        if np.all(eigenvalues > cost_tolerance):
            message = "The optimiser has located a minimum."
        elif np.all(eigenvalues < -cost_tolerance):
            message = "The optimiser has located a maximum."
        elif np.all(np.abs(eigenvalues) > cost_tolerance):
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

    print(message)

    try:
        # Build a plotting span around x. adjust the multiplier if you want a wider/narrower view
        span_multiplier = 4.0
        span0 = (x[0] - span_multiplier * dx[0], x[0] + span_multiplier * dx[0])
        span1 = (x[1] - span_multiplier * dx[1], x[1] + span_multiplier * dx[1])

        ng = 41  # grid resolution per axis
        param0 = np.linspace(span0[0], span0[1], ng)
        param1 = np.linspace(span1[0], span1[1], ng)

        # Evaluate cost on the grid
        Z = np.empty((ng, ng), dtype=float)
        for i in range(ng):
            for j in range(ng):
                p = np.array([param0[i], param1[j]], dtype=float)
                try:
                    Z[i, j] = float(cost(p))
                except Exception:
                    Z[i, j] = np.nan

        # cost contour
        fig, ax = plt.subplots(figsize=(6, 5))
        finiteZ = Z[np.isfinite(Z)]
        if finiteZ.size == 0:
            ax.text(
                0.5,
                0.5,
                "No finite cost values on contour grid",
                ha="center",
                va="center",
            )
        else:
            vmin, vmax = np.nanmin(Z), np.nanmax(Z)
            # number of contour levels
            levels = np.linspace(vmin, vmax, 10)
            # note: Z was filled as Z[i, j] with i along param0 and j along param1,
            # matplotlib's contour expects x,y meshgrid order; using transpose aligns axes.
            cs = ax.contour(param0, param1, Z.T, levels=levels)
            ax.clabel(cs, inline=1, fontsize=8)

        ax.scatter([x[0]], [x[1]], marker="x", s=60, label="optimum")
        ax.set_xlabel(names[0] if names is not None else "param0")
        ax.set_ylabel(names[1] if names is not None else "param1")
        ax.set_title("Cost contour near optimum")
        ax.legend(loc="best")

        # eigenvector figure (over same axes)
        fig_eig, ax_eig = plt.subplots(figsize=(6, 5))
        ax_eig.scatter([x[0]], [x[1]], marker="x", s=60, label="optimum")

        span_param0_len = span0[1] - span0[0]
        span_param1_len = span1[1] - span1[0]
        length = 0.5 * np.sqrt(span_param0_len**2 + span_param1_len**2)

        # Plot eigenvectors
        if eigenvectors is not None and np.isfinite(eigenvectors).all():
            for k in range(eigenvectors.shape[1]):
                v = eigenvectors[:, k].astype(float)
                nrm = np.linalg.norm(v)
                if nrm < 1e-20:
                    continue
                vn = v / nrm
                p_minus = x - vn * length
                p_plus = x + vn * length
                ax_eig.plot(
                    [p_minus[0], p_plus[0]],
                    [p_minus[1], p_plus[1]],
                    linestyle="--",
                    linewidth=1.2,
                    label=f"eig {k} (λ={eigenvalues[k]:.3g})",
                )

        ax_eig.set_xlabel(names[0] if names is not None else "param0")
        ax_eig.set_ylabel(names[1] if names is not None else "param0")
        ax_eig.set_title("Hessian eigenvectors")
        ax_eig.legend(loc="best")

        # try to set same limits as contour for easy comparison
        try:
            ax.set_xlim(span0[0], span0[1])
            ax.set_ylim(span1[0], span1[1])
            ax_eig.set_xlim(span0[0], span0[1])
            ax_eig.set_ylim(span1[0], span1[1])
        except Exception:
            pass

        plt.show()

    except Exception as e_plot:
        try:
            print("Plotting failed:", e_plot)
        except Exception:
            pass

    return message
