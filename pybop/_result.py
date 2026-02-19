import json
import pickle

import numpy as np
import pandas as pd
from scipy.io import savemat

from pybop import plot
from pybop._logging import Logger
from pybop.problems.problem import Problem


class NumpyEncoder(json.JSONEncoder):
    """
    Numpy serialiser helper class that converts numpy arrays to a list.
    Numpy arrays cannot be directly converted to JSON, so the arrays are
    converted to python list objects before encoding.
    """

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # won't be called since we only need to convert numpy arrays
        return json.JSONEncoder.default(self, obj)  # pragma: no cover


class Result:
    """
    Stores the result produced by an optimiser or sampler.

    Attributes
    ----------
    problem : pybop.Problem
        The optimisation problem used to generate the results.
    logger : pybop.Logger
        The log of the optimisation or sampling process.
    time : float
        The time taken.
    method_name : str
        The name of the optimiser or sampler.
    message : str
        The reason for stopping given by the optimiser or sampler.
    """

    def __init__(
        self,
        problem: Problem,
        logger: Logger,
        time: float,
        method_name: str | None = None,
        message: str | None = None,
        scipy_result=None,
    ):
        self._problem = problem
        self._minimising = problem.minimising
        self.method_name = method_name
        self.n_runs = 0
        self._best_run = None
        self._x = [logger.x_model_best]
        self._x_model = [logger.x_model]
        self._x0 = [logger.x0]
        self._best_cost = [logger.cost_best]
        self._cost = [logger.cost_convergence]
        self._initial_cost = [logger.cost[0]]
        self._n_iterations = [logger.iteration]
        self._iteration_number = [logger.iteration_number]
        self._n_evaluations = [logger.evaluations]
        self._message = [message]
        self._scipy_result = [scipy_result]
        self._time = [time]

        self._validate()

    @staticmethod
    def combine(results: list["Result"]) -> "Result":
        """
        Combine multiple Result objects into a single one.

        Parameters
        ----------
        results : list[Result]
            List of Result objects to combine.

        Returns
        -------
        Result
            Combined Result object.
        """
        if len(results) == 0:
            raise ValueError("No results to combine.")
        ret = results[0]
        ret._x = [x for result in results for x in result._x]  # noqa: SLF001
        ret._x_model = [x for result in results for x in result._x_model]  # noqa: SLF001
        ret._x0 = [x for result in results for x in result._x0]  # noqa: SLF001
        ret._best_cost = [  # noqa: SLF001
            x
            for result in results
            for x in result._best_cost  # noqa: SLF001
        ]
        ret._cost = [x for result in results for x in result._cost]  # noqa: SLF001
        ret._initial_cost = [  # noqa: SLF001
            x
            for result in results
            for x in result._initial_cost  # noqa: SLF001
        ]
        ret._n_iterations = [  # noqa: SLF001
            x
            for result in results
            for x in result._n_iterations  # noqa: SLF001
        ]
        ret._iteration_number = [  # noqa: SLF001
            x
            for result in results
            for x in result._iteration_number  # noqa: SLF001
        ]
        ret._n_evaluations = [  # noqa: SLF001
            x
            for result in results
            for x in result._n_evaluations  # noqa: SLF001
        ]
        ret._message = [  # noqa: SLF001
            x
            for result in results
            for x in result._message  # noqa: SLF001
        ]
        ret._scipy_result = [  # noqa: SLF001
            x
            for result in results
            for x in result._scipy_result  # noqa: SLF001
        ]
        ret._time = [x for result in results for x in result._time]  # noqa: SLF001

        ret._best_run = None  # noqa: SLF001
        ret.n_runs = len(results)
        ret._validate()  # noqa: SLF001

        return ret

    def _validate(self):
        """Check that there is a finite cost and update best run."""
        self._check_for_finite_cost()
        if self._minimising:
            self._best_run = self._best_cost.index(min(self._best_cost))
        else:
            self._best_run = self._best_cost.index(max(self._best_cost))

    def _check_for_finite_cost(self) -> None:
        """
        Validate the optimised parameters and ensure they produce a finite cost value.

        Raises:
            ValueError: If the optimised parameters do not produce a finite cost value.
        """
        if not any(np.isfinite(self._best_cost)):
            raise ValueError(
                f"Optimised parameters {self._problem.parameters.to_dict(self._x[-1])} do not produce a finite cost value."
            )

    def __str__(self) -> str:
        """
        A string representation of the Result object.

        Returns:
            str: A formatted string containing optimisation result information.
        """
        return (
            f"Result:\n"
            f"  Best result from {self.n_runs} run(s).\n"
            f"  Initial parameters: {self.x0}\n"
            f"  Optimised parameters: {self.x}\n"
            f"  Best cost: {self.best_cost}\n"
            f"  Optimisation time: {self.time} seconds\n"
            f"  Number of iterations: {self.total_iterations()}\n"
            f"  Number of evaluations: {self.total_evaluations()}\n"
            f"  Reason for stopping: {self.message}"
        )

    def total_iterations(self) -> np.floating | None:
        """Calculates the total number of iterations across all runs."""
        return np.sum(self._n_iterations) if len(self._n_iterations) > 0 else None

    def total_evaluations(self) -> np.floating | None:
        """Calculates the total number of evaluations across all runs."""
        return np.sum(self._n_evaluations) if len(self._n_evaluations) > 0 else None

    def total_runtime(self) -> np.floating | None:
        """Calculates the total runtime across all runs."""
        return np.sum(self._time) if len(self._time) > 0 else None

    def _get_single_or_all(self, attr):
        value = getattr(self, attr)
        if len(value) > 1:
            return value[self._best_run]
        return value[0]

    @property
    def x(self) -> np.ndarray:
        """The best parameter values (in model space)."""
        return self._get_single_or_all("_x")

    @property
    def x_model(self) -> np.ndarray:
        """The log of the evaluated parameters (in model space)."""
        return self._get_single_or_all("_x_model")

    @property
    def x0(self) -> np.ndarray:
        """The initial parameter values."""
        return self._get_single_or_all("_x0")

    @property
    def best_inputs(self) -> dict[str, np.ndarray]:
        """The best parameters as a dictionary."""
        return self._problem.parameters.to_dict(self.x)

    @property
    def best_cost(self) -> float:
        """The best cost value(s)."""
        return self._get_single_or_all("_best_cost")

    @property
    def cost(self) -> np.ndarray:
        """The log of the cost values."""
        return self._get_single_or_all("_cost")

    @property
    def initial_cost(self) -> float:
        """The initial cost value(s)."""
        return self._get_single_or_all("_initial_cost")

    @property
    def n_iterations(self) -> int:
        """The number of iterations."""
        return self._get_single_or_all("_n_iterations")

    @property
    def iteration_number(self) -> np.ndarray | None:
        """The number of iterations."""
        return self._get_single_or_all("_iteration_number")

    @property
    def n_evaluations(self) -> int:
        """The number of evaluations."""
        return self._get_single_or_all("_n_evaluations")

    @property
    def problem(self) -> Problem:
        """The optimisation problem."""
        return self._problem

    @property
    def minimising(self) -> bool:
        """Whether the cost was minimised (or maximised)."""
        return self._minimising

    @property
    def message(self) -> str | None:
        """The optimisation termination message(s)."""
        return self._get_single_or_all("_message")

    @property
    def scipy_result(self):
        """The SciPy result."""
        return self._get_single_or_all("_scipy_result")

    @property
    def time(self) -> float | None:
        """The optimisation time(s)."""
        return self.total_runtime()

    def plot_convergence(self, **kwargs):
        """
        Plot the evolution of the best cost during the optimisation.

        Parameters
        ----------
        show : bool, optional
            If True, the figure is shown upon creation (default: True).
        **layout_kwargs : optional
            Valid Plotly layout keys and their values.
        """
        return plot.convergence(result=self, **kwargs)

    def plot_parameters(self, **kwargs):
        """
        Plot the evolution of parameter values during the optimisation.

        Parameters
        ----------
        show : bool, optional
            If True, the figure is shown upon creation (default: True).
        **layout_kwargs : optional
            Valid Plotly layout keys and their values.
        """
        return plot.parameters(result=self, **kwargs)

    def plot_surface(self, **kwargs):
        """
        Plot a 2D representation of the Voronoi diagram with color-coded regions.

        Parameters
        ----------
        bounds : numpy.ndarray, optional
            A 2x2 array specifying the [min, max] bounds for each parameter.
        normalise : bool, optional
            If True, the voronoi regions are computed using the Euclidean distance between
            points normalised with respect to the bounds (default: True).
        resolution : int, optional
            Resolution of the plot (default: 500).
        show : bool, optional
            If True, the figure is shown upon creation (default: True).
        **layout_kwargs : optional
            Valid Plotly layout keys and their values.
        """
        return plot.surface(result=self, **kwargs)

    def plot_contour(self, **kwargs):
        """
        Generate and plot a 2D visualisation of the cost landscape with the optimisation trace.

        Parameters
        ----------
        gradient : bool, optional
            If True, gradient plots are also generated (default: False).
        bounds : numpy.ndarray | list[list[float]], optional
            A 2x2 array specifying the [min, max] bounds for each parameter.
        transformed : bool, optional
            Uses the transformed parameter values, as seen by the optimiser (default: False).
        steps : int, optional
            The number of grid points to divide the parameter space into along each dimension
            (default: 10).
        show : bool, optional
            If True, the figure is shown upon creation (default: True).
        **layout_kwargs : optional
            Valid Plotly layout keys and their values.
        """
        return plot.contour(call_object=self, **kwargs)

    def save(self, filename):
        """Save the whole result using pickle"""

        with open(filename, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def data_dict(self, short_names):
        data = {}
        for key, value in self.best_inputs.items():
            if short_names is not None and key in short_names.keys():
                data[short_names[key]] = value
            else:
                data[key] = value

        return data

    def save_data(
        self, filename=None, variables=None, to_format="pickle", short_names=None
    ):
        """
        Save result data (raw arrays)

        Based on pybamm.Solution.save_data

        Parameters
        ----------
        filename : str, optional
            The name of the file to save data to. If None, then a str is returned
        to_format : str, optional
            The format to save to. Options are:

            - 'pickle' (default): creates a pickle file with the data dictionary
            - 'matlab': creates a .mat file, for loading in matlab
            - 'csv': creates a csv file (0D variables only)
            - 'json': creates a json file
        short_names : dict, optional
            Dictionary of shortened names to use when saving. This may be necessary when
            saving to MATLAB, since no spaces or special characters are allowed in
            MATLAB variable names. Note that not all the variables need to be given
            a short name.

        Returns
        -------
        data : str, optional
            str if 'csv' or 'json' is chosen and filename is None, otherwise None
        """

        data = self.data_dict(short_names)

        if to_format == "pickle":
            if filename is None:
                raise ValueError("pickle format must be written to a file")
            with open(filename, "wb") as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        elif to_format == "matlab":
            if filename is None:
                raise ValueError("matlab format must be written to a file")
            # Check all the variable names only contain a-z, A-Z or _ or numbers
            for name in data.keys():
                # Check the string only contains the following ASCII:
                # a-z (97-122)
                # A-Z (65-90)
                # _ (95)
                # 0-9 (48-57) but not in the first position
                for i, s in enumerate(name):
                    if not (
                        97 <= ord(s) <= 122
                        or 65 <= ord(s) <= 90
                        or ord(s) == 95
                        or (i > 0 and 48 <= ord(s) <= 57)
                    ):
                        raise ValueError(
                            f"Invalid character '{s}' found in '{name}'. "
                            "MATLAB variable names must only contain a-z, A-Z, _, "
                            "or 0-9 (except the first position). "
                            "Use the 'short_names' argument to pass an alternative "
                            "variable name, e.g. \n\n"
                            "\tsolution.save_data(filename, "
                            "['Electrolyte concentration'], to_format='matlab, "
                            "short_names={'Electrolyte concentration': 'c_e'})"
                        )
            savemat(filename, data)
        elif to_format == "csv":
            for name, var in data.items():
                if var.ndim == 0:
                    data[name] = [var]
                elif var.ndim >= 2:
                    raise ValueError(
                        f"only 0D variables can be saved to csv, but '{name}' is {var.ndim - 1}D"
                    )
            df = pd.DataFrame(data)
            return df.to_csv(filename, index=False)
        elif to_format == "json":
            if filename is None:
                return json.dumps(data, cls=NumpyEncoder)
            else:
                with open(filename, "w") as outfile:
                    json.dump(data, outfile, cls=NumpyEncoder)
        else:
            raise ValueError(f"format '{to_format}' not recognised")
