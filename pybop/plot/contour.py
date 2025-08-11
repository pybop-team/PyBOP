import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.interpolate import griddata

from pybop import BaseOptimiser, Problem
from pybop.plot.plotly_manager import PlotlyManager


@dataclass
class ContourConfig:
    """Container for contour config"""

    gradient: bool = (False,)
    bounds: np.ndarray | None = (None,)
    apply_transform: bool = (False,)
    steps: int = (10,)
    show: bool = (True,)
    use_optim_log: bool = (False,)


@dataclass
class PlotData:
    """Container for plot data."""

    x: np.ndarray
    y: np.ndarray
    costs: np.ndarray
    bounds: np.ndarray
    parameter_names: list[str]
    gradients: np.ndarray | None = None


class ContourPlotter:
    """
    A class for contour plots.
    """

    def __init__(self, problem: Problem, optim: BaseOptimiser = None):
        self.problem = problem
        self.optim = optim
        self.params = self.problem.params
        self.parameter_names = list(self.params.keys())
        self._additional_params = []
        self._parameter_objects_cache = None
        self._validate_parameters()

    def _validate_parameters(
        self,
    ):
        """Validate parameter dimensions"""
        if len(self.params) < 2:
            raise ValueError("This problem takes fewer than 2 parameters.")

        if len(self.params) > 2:
            warnings.warn(
                f"Problem has {len(self.params)} parameters. "
                "Plotting in 2d with fixed values for the additional parameters.",
                UserWarning,
                stacklevel=2,
            )
            self._log_fixed_params()

    def _log_fixed_params(self):
        """Log fixed parameters."""
        for i, param in enumerate(self.params):
            if i > 1:
                print(f"Fixed {param.name}:", param.current_value)
            self._additional_params = self.parameter_names[2:]

    def _get_bounds(
        self, bounds_tuple: tuple[tuple[float, float], ...] | None
    ) -> np.ndarray:
        """Get bounds array"""
        if bounds_tuple is not None:
            return np.array(bounds_tuple)
        return self.params.get_bounds_array()

    def _create_parameter_grid(
        self, bounds: np.ndarray, steps: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create a grid of parameter values for evaluation.

        Returns x_values, y_values, and coordinates
        """
        x_values = np.linspace(bounds[0, 0], bounds[0, 1], steps)
        y_values = np.linspace(bounds[1, 0], bounds[1, 1], steps)
        vals = [self.params[param].current_value for param in self._additional_params]

        # Create a mesh and append fixed parameters to generate coords.
        x_mesh, y_mesh = np.meshgrid(x_values, y_values, indexing="ij")
        vals = [np.ones(x_mesh.size) * val for val in vals]
        coordinates = np.stack([x_mesh.ravel(), y_mesh.ravel(), *vals], axis=1)

        return x_values, y_values, coordinates

    def _evaluate_cost_function(
        self, coordinates: np.ndarray, compute_gradients: bool
    ) -> tuple[np.ndarray, list[np.ndarray] | None]:
        """Evaluate the cost function over the parameter grid."""
        self.problem.set_params(coordinates)

        # Todo: transform gradient with corresponding parameter transformations
        if compute_gradients:
            costs, grad = self.problem.run_with_sensitivities()
            return costs, grad
        else:
            costs = self.problem.run()
            return costs, None

    def _interpolate_with_optimisation_log(
        self, x_values: np.ndarray, y_values: np.ndarray, costs: np.ndarray, steps: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Interpolate cost surface using optimisation log data."""
        if self.optim is None or not hasattr(self.optim, "log"):
            return x_values, y_values, costs

        # Pre-allocate arrays
        param_log = np.asarray(self.optim.log.x_model)

        # Create a meshgrid
        x_indices, y_indices = np.meshgrid(
            np.arange(steps), np.arange(steps), indexing="ij"
        )
        flat_x = x_values[x_indices.ravel()]
        flat_y = y_values[y_indices.ravel()]
        flat_costs = costs.ravel()

        # Concatenate with the optimiser log
        combined_x = np.concatenate([flat_x, param_log[:, 0]])
        combined_y = np.concatenate([flat_y, param_log[:, 1]])
        combined_costs = np.concatenate([flat_costs, self.optim.log.cost])

        # Create a mesh for interpolating costs
        unique_x = np.unique(combined_x)
        unique_y = np.unique(combined_y)
        mesh_x, mesh_y = np.meshgrid(unique_x, unique_y, indexing="ij")

        interpolated_costs = griddata(
            (combined_x, combined_y), combined_costs, (mesh_x, mesh_y), method="linear"
        )

        return unique_x, unique_y, interpolated_costs

    def _get_parameter_objects(self) -> tuple:
        """Get parameter objects with caching."""
        if self._parameter_objects_cache is None:
            self._parameter_objects_cache = (
                self.params[self.parameter_names[0]],
                self.params[self.parameter_names[1]],
            )
        return self._parameter_objects_cache

    def _apply_parameter_transformation_vectorised(
        self, values: np.ndarray, parameter_index: int, apply_transform: bool
    ) -> np.ndarray:
        """Apply parameter transformation using vectorised operations."""
        if not apply_transform:
            return values

        param_obj = self._get_parameter_objects()[parameter_index]

        # List comprehension for common dimension
        if len(values) == 1:
            return np.array([param_obj.transformation.to_search(values[0])])

        # Alternative vectorized implementation
        vectorised_transform = np.vectorize(param_obj.transformation.to_search)
        return vectorised_transform(values)

    def _prepare_plot_data(self, config: ContourConfig) -> PlotData:
        """Prepare all data needed for plotting."""
        # Convert bounds to tuple for caching if provided
        bounds_tuple = None
        if config.bounds is not None:
            bounds_tuple = tuple(tuple(row) for row in config.bounds)

        bounds = self._get_bounds(bounds_tuple)

        # Create parameter grid
        x_values, y_values, coordinates = self._create_parameter_grid(
            bounds, config.steps
        )

        # Evaluate cost function and reshape
        costs, gradients = self._evaluate_cost_function(coordinates, config.gradient)
        costs = costs.reshape(
            config.steps, config.steps, order="F"
        )  # Column-major reshape

        # Apply optimisation log interpolation if requested
        if config.use_optim_log and self.optim is not None:
            x_values, y_values, costs = self._interpolate_with_optimisation_log(
                x_values, y_values, costs, config.steps
            )

        x_transformed = self._apply_parameter_transformation_vectorised(
            x_values, 0, config.apply_transform
        )
        y_transformed = self._apply_parameter_transformation_vectorised(
            y_values, 1, config.apply_transform
        )

        # Transform bounds
        bounds_transformed = bounds.copy()
        if config.apply_transform:
            bounds_transformed[0] = self._apply_parameter_transformation_vectorised(
                bounds[0], 0, True
            )
            bounds_transformed[1] = self._apply_parameter_transformation_vectorised(
                bounds[1], 1, True
            )

        # Process gradients
        processed_gradients = None
        if gradients is not None:
            target_shape = (config.steps, config.steps)
            processed_gradients = [
                grad.reshape(target_shape)
                for grad in np.hsplit(gradients, len(self.params))
            ]

        return PlotData(
            x=x_transformed,
            y=y_transformed,
            costs=costs,
            bounds=bounds_transformed,
            parameter_names=self.parameter_names,
            gradients=processed_gradients,
        )

    def _create_base_layout(
        self, plot_data: PlotData, config: ContourConfig
    ) -> dict[str, Any]:
        """Create the base layout configuration for plots."""
        names = plot_data.parameter_names
        x_label = f"Transformed {names[0]}" if config.apply_transform else names[0]
        y_label = f"Transformed {names[1]}" if config.apply_transform else names[1]

        return {
            "title": "Cost Landscape",
            "title_x": 0.5,
            "title_y": 0.905,
            "width": 600,
            "height": 600,
            "xaxis": {
                "title": x_label,
                "range": plot_data.bounds[
                    0
                ].tolist(),  # Convert to list for JSON serialisation
                "showexponent": "last",
                "exponentformat": "e",
            },
            "yaxis": {
                "title": y_label,
                "range": plot_data.bounds[1].tolist(),
                "showexponent": "last",
                "exponentformat": "e",
            },
            "legend": {
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1,
                "xanchor": "right",
                "x": 1,
            },
        }

    def _extract_optimisation_data(self) -> tuple:
        """Extract optimisation data once for reuse."""
        if self.optim is None or not hasattr(self.optim, "log"):
            return None, None, None

        log = self.optim.log

        # Extract trace data
        trace_data = None
        if hasattr(log, "x_model") and log.x_model:
            trace_array = np.asarray([item[:2] for item in log.x_model])
            trace_data = trace_array.reshape(-1, 2)

        # Extract initial guess
        initial_guess = None
        if hasattr(log, "x0") and log.x0 is not None:
            initial_guess = np.array(log.x0[:2])

        # Extract final values
        final_values = None
        if hasattr(log, "last_x_model_best") and log.last_x_model_best is not None:
            final_values = np.array(log.last_x_model_best[:2])

        return trace_data, initial_guess, final_values

    def _add_optimisation_traces(self, fig, config: ContourConfig) -> None:
        """Add optimisation traces to the figure."""
        trace_data, initial_guess, final_values = self._extract_optimisation_data()

        if trace_data is None and initial_guess is None and final_values is None:
            return

        go = PlotlyManager().go

        # Add optimisation path
        if trace_data is not None:
            x_trace = self._apply_parameter_transformation_vectorised(
                trace_data[:, 0], 0, config.apply_transform
            )
            y_trace = self._apply_parameter_transformation_vectorised(
                trace_data[:, 1], 1, config.apply_transform
            )

            # Pre-compute color scale
            n_points = len(trace_data)
            colors = np.linspace(0, 1, n_points)

            fig.add_trace(
                go.Scatter(
                    x=x_trace,
                    y=y_trace,
                    mode="markers",
                    marker=dict(
                        color=colors,
                        colorscale="Greys",
                        size=8,
                        showscale=False,
                    ),
                    showlegend=False,
                )
            )

        # Add initial guess marker
        if initial_guess is not None:
            x0_transformed = self._apply_parameter_transformation_vectorised(
                initial_guess[0:1], 0, config.apply_transform
            )
            y0_transformed = self._apply_parameter_transformation_vectorised(
                initial_guess[1:2], 1, config.apply_transform
            )

            fig.add_trace(
                go.Scatter(
                    x=x0_transformed,
                    y=y0_transformed,
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

        # Add final optimised value marker
        if final_values is not None:
            final_x_transformed = self._apply_parameter_transformation_vectorised(
                final_values[0:1], 0, config.apply_transform
            )
            final_y_transformed = self._apply_parameter_transformation_vectorised(
                final_values[1:2], 1, config.apply_transform
            )

            fig.add_trace(
                go.Scatter(
                    x=final_x_transformed,
                    y=final_y_transformed,
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

    def _create_gradient_figures(
        self,
        plot_data: PlotData,
        base_layout: dict[str, Any],
        **layout_kwargs,
    ) -> list[Any]:
        """Create gradient figures if gradients are available."""
        if plot_data.gradients is None:
            return []

        go = PlotlyManager().go
        gradient_figures = []

        # Pre-create layout copies
        for i, gradient_data in enumerate(plot_data.gradients):
            layout = base_layout.copy()
            layout["title"] = f"Gradient for Parameter: {i + 1}"

            fig = go.Figure(
                data=[
                    go.Contour(
                        x=plot_data.x,
                        y=plot_data.y,
                        z=gradient_data,
                        colorscale="Viridis",
                        connectgaps=True,
                    )
                ],
                layout=go.Layout(layout),
            )

            fig.update_layout(**layout_kwargs)
            gradient_figures.append(fig)

        return gradient_figures

    def create_contour_plot(
        self, config: ContourConfig, **layout_kwargs
    ) -> Any | tuple[Any, list[Any]]:
        """Create contour plot with the given configuration."""
        plot_data = self._prepare_plot_data(config)
        go = PlotlyManager().go

        # Create base layout
        base_layout = self._create_base_layout(plot_data, config)

        # Create base contour figure
        fig = go.Figure(
            data=[
                go.Contour(
                    x=plot_data.x,
                    y=plot_data.y,
                    z=plot_data.costs,
                    colorscale="Viridis",
                    connectgaps=False,
                )
            ],
            layout=go.Layout(base_layout),
        )

        # Add optimisation traces if available
        self._add_optimisation_traces(fig, config)

        # Apply custom layout options, show fig
        fig.update_layout(**layout_kwargs)

        if config.show:
            fig.show()

        # Create gradient figures if requested
        if config.gradient:
            gradient_figures = self._create_gradient_figures(
                plot_data, base_layout, **layout_kwargs
            )

            if config.show:
                for grad_fig in gradient_figures:
                    grad_fig.show()

            return fig, gradient_figures

        return fig


def contour(
    call_object: Problem | BaseOptimiser,
    gradient: bool = False,
    bounds: np.ndarray | None = None,
    apply_transform: bool = False,
    steps: int = 10,
    show: bool = True,
    use_optim_log: bool = False,
    **layout_kwargs,
) -> Any | tuple[Any, list[Any]]:
    """
    Plot a 2D visualisation of a cost landscape using Plotly.

    This function generates a contour plot representing the cost landscape for a provided
    callable cost function over a grid of parameter values within the specified bounds.

    Parameters
    ----------
    call_object : Problem | BaseOptimiser
        Either a pybop.Problem object or a pybop.BaseOptimiser object which provides
        a specific optimisation trace overlaid on the cost landscape.
    gradient : bool, optional
        If True, gradient plots are also generated (default: False).
    bounds : np.ndarray, optional
        A 2x2 array specifying the [min, max] bounds for each parameter. If None, uses
        the parameter bounds from the problem.
    apply_transform : bool, optional
        Uses the transformed parameter values (as seen by the optimiser) for plotting. (Default: False)
    steps : int, optional
        The number of grid points to divide the parameter space into along each dimension (default: 10).
    show : bool, optional
        If True, the figure is shown upon creation (default: True).
    use_optim_log : bool, optional
        If True, the optimisation log is used to overlay the optimiser convergence trace (default: False).
    **layout_kwargs : optional
        Valid Plotly layout keys and their values.

    Returns
    -------
    plotly.graph_objs.Figure | tuple[plotly.graph_objs.Figure, list[plotly.graph_objs.Figure]]]
        The Plotly figure object containing the cost landscape plot.
        If gradient=True, returns a tuple of (main_figure, gradient_figures).

    Raises
    ------
    TypeError
        If call_object is not a Problem or BaseOptimiser instance.
    ValueError
        If the problem has fewer than 2 parameters or if steps <= 0.
    """
    # Extract problem and optimiser from call_object
    if isinstance(call_object, BaseOptimiser):
        problem = call_object.problem
        optimiser = call_object
    elif isinstance(call_object, Problem):
        problem = call_object
        optimiser = None
    else:
        raise TypeError(
            "call_object must be a pybop.Problem or pybop.BaseOptimiser instance."
        )

    # Create configuration
    config = ContourConfig(
        gradient=gradient,
        bounds=bounds,
        apply_transform=apply_transform,
        steps=steps,
        show=show,
        use_optim_log=use_optim_log,
    )

    # Create plotter and generate plot
    plotter = ContourPlotter(problem, optimiser)
    return plotter.create_contour_plot(config, **layout_kwargs)
