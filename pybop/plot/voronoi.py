from typing import Union

import numpy as np
from scipy.spatial import Voronoi, cKDTree

from pybop import BaseOptimiser, Optimisation
from pybop.plot.plotly_manager import PlotlyManager


def _voronoi_regions(x, y, f, xlim, ylim):
    """
    Takes a set of ``(x, y, f)`` points and returns the edgepoints of the
    voronoi region around each point within the boundaries specified by
    ``xlim`` and ``ylim``. Originally adapted from PINTS.

    Parameters
    ----------
    x : array-like
        A list of x-coordinates.
    y : array-like
        A list of y-coordinates.
    f : array-like
        The score function at the given x and y coordinates.
    xlim : tuple
        Lower and upper bound for the x coordinates.
    ylim : tuple
        Lower and upper bound for the y coordinates.

    Returns
    -------
    tuple
        A tuple ``(x, y, f, regions)`` where ``x``, ``y`` and ``f`` are the
        coordinates of the accepted points and each ``regions[i]`` is a list of the
        vertices making up the voronoi region for point ``(x[i], y[i])`` with score
        ``f[i]``.
    """
    # Convert inputs to numpy arrays
    x, y, f = map(np.array, (x, y, f))

    # Check limits
    xmin, xmax = sorted(map(float, xlim))
    ymin, ymax = sorted(map(float, ylim))

    # Create voronoi diagram
    vor = Voronoi(np.column_stack((x, y)))

    # Calculate center and radius
    center = vor.points.mean(axis=0)
    radius2 = 2 * np.sqrt((xmax - xmin) ** 2 + (ymax - ymin) ** 2)

    # Create regions
    regions = [set() for _ in range(len(x))]
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        v1, v2 = sorted([v1, v2])  # Sort the vertices
        x2 = vor.vertices[v2]
        y1, y2 = vor.points[p1], vor.points[p2]
        x1 = (
            vor.vertices[v1]
            if v1 >= 0
            else _compute_far_point(vor, y1, y2, x2, center, radius2)
        )
        regions[p1].update((tuple(x1), tuple(x2)))
        regions[p2].update((tuple(x1), tuple(x2)))

    # Process regions
    selection = []
    for k, region in enumerate(regions):
        # Check for empty regions
        if not region:
            continue  # pragma: no cover

        # Check for regions completely outside of limits, skip if so.
        region = np.asarray(list(region))
        xmn, xmx, ymn, ymx, all_outside = _is_region_outside_bounds(
            region, xmin, xmax, ymin, ymax
        )

        if all_outside:
            continue  # pragma: no cover

        # Sort the vertices
        regions[k] = _sort_region_vertices(region, vor.points[k])

        # Region fully contained? Then keep in selection and continue
        if not (np.any(xmn) or np.any(xmx) or np.any(ymn) or np.any(ymx)):
            selection.append(k)
            continue

        # Truncate points for x-axis and y-axis
        region = truncate_region_by_axis(region, 0, xmin, xmax)
        region = truncate_region_by_axis(region, 1, ymin, ymax)

        # Filter region if len > 2
        if len(region) > 2:
            selection.append(k)

    # Filter out bad regions
    regions = [regions[i] for i in selection]
    x, y, f = x[selection], y[selection], f[selection]

    return x, y, f, regions


def _compute_far_point(vor, y1, y2, x2, center, radius2):
    """
    Computes the far-point of the voronoi region.
    Originally adapted from PINTS.
    """
    t = y2 - y1
    t /= np.linalg.norm(t)
    q = np.array([-t[1], t[0]])
    midpoint = np.mean([y1, y2], axis=0)
    return x2 + np.sign(np.dot(midpoint - center, q)) * q * radius2


def _is_region_outside_bounds(region, xmin, xmax, ymin, ymax):
    """
    Check for points outside of bounds. Originally adapted from PINTS.
    """
    xmn = region[:, 0] < xmin
    xmx = region[:, 0] > xmax
    ymn = region[:, 1] < ymin
    ymx = region[:, 1] > ymax
    combined = np.all(xmn) or np.all(xmx) or np.all(ymn) or np.all(ymx)
    return xmn, xmx, ymn, ymx, combined


def _sort_region_vertices(region, point):
    """
    Sorts the region vertices according to the given point.
    Originally adapted from PINTS.
    """
    angles = np.arctan2(region[:, 1] - point[1], region[:, 0] - point[0])
    return region[np.argsort(angles)]


def truncate_region_by_axis(region, axis, min_val, max_val):
    """
    Truncates a region along a specific axis. Originally adapted from PINTS.

    Args:
        region: A list of points (numpy arrays) representing the region.
        axis: The axis to truncate along (0 for x, 1 for y).
        min_val: The minimum value along the axis.
        max_val: The maximum value along the axis.

    Returns:
        A list of points representing the truncated region.
    """

    new_region = []
    for j, p in enumerate(region):
        q = region[j - 1] if j > 0 else region[-1]
        r = region[j + 1] if j < len(region) - 1 else region[0]

        if p[axis] < min_val:
            if q[axis] < min_val and r[axis] < min_val:
                continue
            if q[axis] >= min_val:
                new_region.append(interpolate_point(p, q, axis, min_val))
            if r[axis] >= min_val:
                new_region.append(interpolate_point(p, r, axis, min_val))
        elif p[axis] > max_val:
            if q[axis] > max_val and r[axis] > max_val:
                continue  # pragma: no cover
            if q[axis] <= max_val:
                new_region.append(interpolate_point(p, q, axis, max_val))
            if r[axis] <= max_val:
                new_region.append(interpolate_point(p, r, axis, max_val))
        else:
            new_region.append(p)

    return new_region


def interpolate_point(p, q, axis, boundary_val):
    """
    Interpolates a new point on the line segment between p and q at the given boundary value.

    Args:
        p: The first point.
        q: The second point.
        axis: The axis to interpolate along (0 for x, 1 for y).
        boundary_val: The boundary value.

    Returns:
        A numpy array representing the interpolated point.
    """

    other_axis = 1 - axis
    s = p[other_axis] + (boundary_val - p[axis]) * (q[other_axis] - p[other_axis]) / (
        q[axis] - p[axis]
    )
    return np.array([boundary_val, s]) if axis == 0 else np.array([s, boundary_val])


def assign_nearest_value(x, y, f, xi, yi):
    """
    Computes an array of values given by the score of the nearest point.

    Parameters
    ----------
    x : array-like
        The x coordinates of points with known scores.
    y : array-like
        The y coordinates of points with known scores.
    f : array-like
        The score function at the given x and y coordinates.
    xi : array-like
        The x coordinates of grid points.
    yi : array-like
        The y coordinates of grid points.

    Returns
    -------
        A numpy array containing the scores corresponding to the grid points.
    """
    # Create a KD-tree for efficient nearest neighbor search
    tree = cKDTree(np.column_stack((x, y)))

    # Find the nearest point for each grid point
    _, indices = tree.query(np.column_stack((xi.ravel(), yi.ravel())))
    zi = f[indices].reshape(xi.shape)

    return zi


def surface(
    optim: Union[BaseOptimiser, Optimisation],
    bounds=None,
    normalise=True,
    resolution=250,
    show=True,
    **layout_kwargs,
):
    """
    Plot a 2D representation of the Voronoi diagram with color-coded regions.

    Parameters:
    -----------
    optim : pybop.BaseOptimiser | pybop.Optimisation
        Solved optimisation object
    bounds : numpy.ndarray, optional
        A 2x2 array specifying the [min, max] bounds for each parameter. If None, uses
        `cost.parameters.get_bounds_for_plotly`.
    normalise : bool, optional
        If True, the voronoi regions are computed using the Euclidean distance between
        points normalised with respect to the bounds (default: True).
    resolution : int, optional
        Resolution of the plot. Default is 500.
    show : bool, optional
        If True, the figure is shown upon creation (default: True).
    **layout_kwargs : optional
        Valid Plotly layout keys and their values,
        e.g. `xaxis_title="Time [s]"` or
        `xaxis={"title": "Time [s]", font={"size":14}}`
    """

    # Append the optimisation trace to the data
    points = optim.log.x

    if points[0].shape[0] != 2:
        raise ValueError("This plot method requires two parameters.")

    x_optim, y_optim = map(list, zip(*points))
    f = optim.log.cost

    # Translate bounds, taking only the first two elements
    xlim, ylim = (
        bounds if bounds is not None else [param.bounds for param in optim.parameters]
    )[:2]

    # Create a grid for plot
    xi = np.linspace(xlim[0], xlim[1], resolution)
    yi = np.linspace(ylim[0], ylim[1], resolution)
    xi, yi = np.meshgrid(xi, yi)

    if normalise:
        if xlim[1] <= xlim[0] or ylim[1] <= ylim[0]:
            raise ValueError("Lower bounds must be strictly less than upper bounds.")

        # Normalise the region
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]
        norm_x_optim = (np.asarray(x_optim) - xlim[0]) / x_range
        norm_y_optim = (np.asarray(y_optim) - ylim[0]) / y_range

        # Compute regions
        norm_x, norm_y, f, norm_regions = _voronoi_regions(
            norm_x_optim, norm_y_optim, f, (0, 1), (0, 1)
        )

        # Create a normalised grid
        norm_xi = np.linspace(0, 1, resolution)
        norm_xi, norm_yi = np.meshgrid(norm_xi, norm_xi)

        # Assign a value to each point in the grid
        zi = assign_nearest_value(norm_x, norm_y, f, norm_xi, norm_yi)

        # Rescale for plotting
        regions = []
        for norm_region in norm_regions:
            region = np.empty_like(norm_region)
            region[:, 0] = norm_region[:, 0] * x_range + xlim[0]
            region[:, 1] = norm_region[:, 1] * y_range + ylim[0]
            regions.append(region)

    else:
        # Compute regions
        x, y, f, regions = _voronoi_regions(x_optim, y_optim, f, xlim, ylim)

        # Assign a value to each point in the grid
        zi = assign_nearest_value(x, y, f, xi, yi)

    # Calculate the size of each Voronoi region
    region_sizes = np.array([len(region) for region in regions])
    relative_sizes = (region_sizes - region_sizes.min()) / (
        region_sizes.max() - region_sizes.min()
    )

    # Construct figure
    go = PlotlyManager().go
    fig = go.Figure()

    # Heatmap
    fig.add_trace(
        go.Heatmap(
            x=xi[0],
            y=yi[:, 0],
            z=zi,
            colorscale="Viridis",
            zsmooth="best",
        )
    )

    # Add Voronoi edges
    for region, size in zip(regions, relative_sizes):
        x_region = region[:, 0].tolist() + [region[0, 0]]
        y_region = region[:, 1].tolist() + [region[0, 1]]

        fig.add_trace(
            go.Scatter(
                x=x_region,
                y=y_region,
                mode="lines",
                line=dict(color="white", width=0.5 + size * 0.1),
                showlegend=False,
            )
        )

    # Add original points
    fig.add_trace(
        go.Scatter(
            x=x_optim,
            y=y_optim,
            mode="markers",
            marker=dict(
                color=[i / len(x_optim) for i in range(len(x_optim))],
                colorscale="Greys",
                size=8,
                showscale=False,
            ),
            text=[f"f={val:.2f}" for val in f],
            hoverinfo="text",
            showlegend=False,
        )
    )

    # Plot the initial guess
    if optim.x0 is not None:
        fig.add_trace(
            go.Scatter(
                x=[optim.x0[0]],
                y=[optim.x0[1]],
                mode="markers",
                marker_symbol="x",
                marker=dict(
                    color="white",
                    line_color="black",
                    line_width=1,
                    size=14,
                    showscale=False,
                ),
                name="Initial values",
            )
        )

        # Plot optimised value
        if optim.log.x_best is not None:
            fig.add_trace(
                go.Scatter(
                    x=[optim.log.x_best[-1][0]],
                    y=[optim.log.x_best[-1][1]],
                    mode="markers",
                    marker_symbol="cross",
                    marker=dict(
                        color="black",
                        line_color="white",
                        line_width=1,
                        size=14,
                        showscale=False,
                    ),
                    name="Final values",
                )
            )

    names = optim.cost.parameters.keys()
    fig.update_layout(
        title="Voronoi Cost Landscape",
        title_x=0.5,
        title_y=0.905,
        xaxis_title=names[0],
        yaxis_title=names[1],
        width=600,
        height=600,
        xaxis=dict(range=xlim, showexponent="last", exponentformat="e"),
        yaxis=dict(range=ylim, showexponent="last", exponentformat="e"),
        legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=1),
    )
    fig.update_layout(**layout_kwargs)
    if show:
        fig.show()
