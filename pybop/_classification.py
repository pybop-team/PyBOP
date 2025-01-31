from typing import Optional

import numpy as np

from pybop import OptimisationResult


def classify_using_hessian(
    result: OptimisationResult, dx=None, cost_tolerance: Optional[float] = 1e-5
):
    """
    A simple check for parameter correlations based on numerical approximation
    of the Hessian matrix at the optimal point using central finite differences.

    Parameters
    ---------
    result : OptimisationResult
        The PyBOP optimisation results.
    dx : array-like, optional
        An array of small positive values used to check proximity to the parameter
        bounds and as the perturbation distance in the finite difference calculations.
    cost_tolerance : float, optional
        A small positive tolerance used for cost value comparisons (default: 1e-5).
    """
    x = result.x
    dx = np.asarray(dx) if dx is not None else np.maximum(x, 1e-40) * 1e-2
    final_cost = result.final_cost
    cost = result.cost
    parameters = cost.parameters
    minimising = result.minimising

    n = len(x)
    if n != 2 or len(dx) != n:
        raise ValueError(
            "The function classify_using_hessian currently only works in the case "
            "of 2 parameters, and dx must have the same length as x."
        )

    # Get a list of parameter names for use in the output message
    names = list(parameters.keys())

    # Evaluate the cost for a grid of surrounding points
    costs = np.zeros((3, 3, 2))
    for i in np.arange(0, 3):
        for j in np.arange(0, 3):
            if i == j == 1:
                costs[1, 1, 0] = final_cost
                costs[1, 1, 1] = final_cost
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
    if (minimising and np.any(costs < final_cost)) or (
        not minimising and np.any(costs > final_cost)
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
                if np.allclose(final_cost, diagonal_costs, atol=cost_tolerance, rtol=0):
                    message += " There may be a correlation between these parameters."

    print(message)
    return message
