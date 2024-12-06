import numpy as np


def classify_using_Hessian(optim, x=None, epsilon=1e-2):
    """
    A simple check for parameter correlations based on numerical approximation
    of the Hessian matrix at the optimal point using central finite differences.

    Paramters
    ---------
    x : array-like, optional
        The parameters values assumed to be the optimal point (default: optim.result.x).
    epsilon : float, optional
        A small positive value used to check proximity to the parameter bounds and as the
        perturbation distance used in the finite difference calculations (default: 1e-2).
    """

    if x is None:
        x = optim.result.x
        final_cost = optim.result.final_cost
    else:
        final_cost = optim.cost(x)

    n = len(x)
    if n != 2:
        raise ValueError(
            "The function classify_using_Hessian currently only works"
            " in the case of 2 parameters."
        )

    # Get a list of parameter names for use in the output message
    names = list(optim.cost.parameters.keys())

    # Evaluate the cost for a grid of surrounding points
    dx = x * epsilon
    costs = np.zeros((3, 3, 2))
    for i in np.arange(0, 3):
        for j in np.arange(0, 3):
            if i == j == 1:
                costs[1, 1, 0] = final_cost
                costs[1, 1, 1] = final_cost
            costs[i, j, 0] = optim.cost(x + np.multiply([i - 1, j - 1], dx))
            costs[i, j, 1] = optim.cost(x + np.multiply([i - 1, j - 1], 2 * dx))

    print(costs - final_cost)

    # Classify the result
    if (optim.minimising and np.any(costs < final_cost)) or (
        not optim.minimising and np.any(costs > final_cost)
    ):
        message = "The optimiser has not converged to a stationary point."

        # Check proximity to bounds
        bounds = optim.cost.parameters.get_bounds()
        if bounds is not None:
            for i, value in enumerate(x):
                x_range = bounds["upper"][i] - bounds["lower"][i]
                if value > bounds["upper"][i] - epsilon * x_range:
                    message += f" The result is near the upper bound of {names[i]}."

                if value < bounds["lower"][i] + epsilon * x_range:
                    message += f" The result is near the lower bound of {names[i]}."
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

        # Compute the eigenvalues, returned in ascending order
        eigr = np.linalg.eigh(cfd_hessian)
        eigenvalues = eigr.eigenvalues

        # Classify the result
        cost_tolerance = epsilon**2 * final_cost
        if np.all(eigenvalues > cost_tolerance):
            message = "The optimiser has located a minimum."
        elif np.all(eigenvalues < -cost_tolerance):
            message = "The optimiser has located a maximum."
        elif np.all(np.abs(eigenvalues) > cost_tolerance):
            message = "The optimiser has located a saddle point."
        else:
            # At least one eigenvalue is too small to classify with certainty
            if np.all(np.abs(eigenvalues) < cost_tolerance):
                message = "The cost is insensitive to these parameters."
            elif np.allclose(eigr.eigenvectors[0], np.array([1, 0])):
                message = f"The cost is insensitive to {names[0]}."
            elif np.allclose(eigr.eigenvectors[0], np.array([0, 1])):
                message = f"The cost is insensitive to {names[1]}."
            else:
                message = "There may be a correlation between these parameters."

    print(message)
    return message
