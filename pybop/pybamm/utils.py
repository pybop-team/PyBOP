import multiprocessing as mp
import platform
from concurrent.futures import TimeoutError as CFTimeoutError

import pybamm
from pebble import ProcessPool


class RecommendedSolver(pybamm.IDAKLUSolver):
    """A shortcut for creating the PyBaMM solver recommended for optimisation."""

    def __init__(self, output_variables: list[str] | None = None):
        solver_options = {"suppress_algebraic_error": True}

        if platform.system() != "Windows":
            solver_options["num_threads"] = max(1, mp.cpu_count())

        super().__init__(
            on_failure="ignore",
            atol=1e-6,
            rtol=1e-6,
            options=solver_options,
            output_variables=output_variables,
        )


class SafeSolver(pybamm.CasadiSolver):
    """
    A version of PyBaMM's CasadiSolver with a timeout option.

    Additional parameters
    ---------------------
    timeout : float, optional
        If timeout is a positive number, simulations are terminated after timeout
        seconds if not completed successfully within this time. Default is None.
    """

    def __init__(self, timeout: float | None = None, **kwargs):
        super().__init__(**kwargs)
        self.timeout = timeout

    def _integrate(
        self,
        model: pybamm.BaseModel,
        t_eval,
        inputs_list: list[dict] | None = None,
        t_interp=None,
        nproc=1,
    ):
        """
        Solve a DAE model defined by residuals with initial conditions y0.

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The model whose solution to calculate.
        t_eval : numeric type
            The times at which to compute the solution
        inputs_list : list of dict, optional
            Any input parameters to pass to the model when solving
        """

        inputs_list = inputs_list or [{}]

        ninputs = len(inputs_list)
        if ninputs == 1 and self.timeout is None:
            new_solution = self._integrate_single(
                model,
                t_eval,
                inputs_list[0],
                model.y0_list[0],
            )
            new_solutions = [new_solution]
        else:
            with ProcessPool(
                context=mp.get_context(self._mp_context),
                max_workers=nproc or mp.cpu_count(),
            ) as p:
                model_list = [model] * ninputs
                t_eval_list = [t_eval] * ninputs
                y0_list = model.y0_list

                futures = p.map(
                    self._integrate_single,
                    model_list,
                    t_eval_list,
                    inputs_list,
                    y0_list,
                    timeout=self.timeout,
                )
                iterator = futures.result()

                new_solutions = []
                while True:
                    try:
                        new_solutions.append(next(iterator))
                    except StopIteration:
                        break
                    except (TimeoutError, CFTimeoutError) as e:
                        raise pybamm.SolverError(
                            f"Timeout after {e.args[1]:.1f} seconds."
                        ) from e
                    except Exception as e:
                        raise pybamm.SolverError(str(e)) from None

        return new_solutions


class SymbolReplacer:
    """
    Helper class to replace all instances of one or more symbols in an expression tree
    with another symbol, as defined by the dictionary `symbol_replacement_map`
    Originally developed by pybamm: https://github.com/pybamm-team/pybamm

    Parameters
    ----------
    symbol_replacement_map : dict {:class:`pybamm.Symbol` -> :class:`pybamm.Symbol`}
        Map of which symbols should be replaced by which.
    processed_symbols: dict {:class:`pybamm.Symbol` -> :class:`pybamm.Symbol`}, optional
        cached replaced symbols
    process_initial_conditions: bool, optional
        Whether to process initial conditions, default is True
    """

    def __init__(
        self,
        symbol_replacement_map: dict[pybamm.Symbol, pybamm.Symbol],
        processed_symbols: dict[pybamm.Symbol, pybamm.Symbol] | None = None,
        process_initial_conditions: bool = True,
    ):
        self._symbol_replacement_map = symbol_replacement_map
        self._processed_symbols = processed_symbols or {}
        self._process_initial_conditions = process_initial_conditions

    def process_model(self, unprocessed_model, inplace=True):
        """
        Replace all instances of a symbol in a PyBaMM model class.

        Parameters
        ----------
        unprocessed_model : :class:`pybamm.BaseModel`
            Model class to assign parameter values to
        inplace: bool, optional
            If True, replace the parameters in the model in place. Otherwise, return a
            new model with parameter values set (default: True).
        """

        model = unprocessed_model if inplace else unprocessed_model.new_copy()

        for variable, equation in unprocessed_model.rhs.items():
            pybamm.logger.verbose(f"Replacing symbols in {variable!r} (rhs)")
            model.rhs[self.process_symbol(variable)] = self.process_symbol(equation)

        for variable, equation in unprocessed_model.algebraic.items():
            pybamm.logger.verbose(f"Replacing symbols in {variable!r} (algebraic)")
            model.algebraic[self.process_symbol(variable)] = self.process_symbol(
                equation
            )

        for variable, equation in unprocessed_model.initial_conditions.items():
            pybamm.logger.verbose(
                f"Replacing symbols in {variable!r} (initial conditions)"
            )
            if self._process_initial_conditions:
                model.initial_conditions[self.process_symbol(variable)] = (
                    self.process_symbol(equation)
                )
            else:
                model.initial_conditions[self.process_symbol(variable)] = equation

        model.boundary_conditions = self.process_boundary_conditions(unprocessed_model)

        for variable, equation in unprocessed_model.variables.items():
            pybamm.logger.verbose(f"Replacing symbols in {variable!r} (variables)")
            model.variables[variable] = self.process_symbol(equation)

        model.events = self._process_events(unprocessed_model.events)
        pybamm.logger.info(f"Finish replacing symbols in {model.name}")

        return model

    def _process_events(self, events: list) -> list:
        new_events = []
        for event in events:
            pybamm.logger.verbose(f"Replacing symbols in event '{event.name}'")
            new_events.append(
                pybamm.Event(
                    event.name, self.process_symbol(event.expression), event.event_type
                )
            )
        return new_events

    def process_boundary_conditions(self, model):
        """
        Process boundary conditions for a PybaMM model class
        Boundary conditions are dictionaries {"left": left bc, "right": right bc}
        in general, but may be imposed on the tabs (or *not* on the tab) for a
        small number of variables, e.g. {"negative tab": neg. tab bc,
        "positive tab": pos. tab bc "no tab": no tab bc}.
        """
        boundary_conditions = {}
        sides = ["left", "right", "negative tab", "positive tab", "no tab"]
        for variable, bcs in model.boundary_conditions.items():
            processed_variable = self.process_symbol(variable)
            boundary_conditions[processed_variable] = {}

            for side in sides:
                try:
                    bc, typ = bcs[side]
                    pybamm.logger.verbose(
                        f"Replacing symbols in {variable!r} ({side} bc)"
                    )
                    processed_bc = (self.process_symbol(bc), typ)
                    boundary_conditions[processed_variable][side] = processed_bc
                except KeyError as err:
                    # Don't raise if side is not in the boundary conditions
                    if err.args[0] in side:
                        pass
                    # Raise otherwise
                    else:  # pragma: no cover
                        raise KeyError(err) from err

        return boundary_conditions

    def process_symbol(self, symbol):
        """
        This function recurses down the tree, replacing any symbols in
        self._symbol_replacement_map.keys() with their corresponding value

        Parameters
        ----------
        symbol : :class:`pybamm.Symbol`
            The symbol to replace

        Returns
        -------
        :class:`pybamm.Symbol`
            Symbol with all replacements performed
        """
        if symbol in self._processed_symbols:
            return self._processed_symbols[symbol]

        processed_symbol = self._process_symbol(symbol)
        self._processed_symbols[symbol] = processed_symbol
        return processed_symbol

    def _process_symbol(self, symbol: pybamm.Symbol) -> pybamm.Symbol:
        if symbol in self._symbol_replacement_map:
            return self._symbol_replacement_map[symbol]

        if isinstance(symbol, pybamm.BinaryOperator):
            # process children
            new_left = self.process_symbol(symbol.left)
            new_right = self.process_symbol(symbol.right)
            return symbol._binary_new_copy(new_left, new_right)  # noqa: SLF001

        if isinstance(symbol, pybamm.UnaryOperator):
            new_child = self.process_symbol(symbol.child)
            return symbol._unary_new_copy(new_child)  # noqa: SLF001

        if isinstance(symbol, pybamm.Function):
            new_children = [self.process_symbol(child) for child in symbol.children]
            # Return a new copy with the replaced symbols
            return symbol._function_new_copy(new_children)  # noqa: SLF001

        if isinstance(symbol, pybamm.Concatenation):
            new_children = [self.process_symbol(child) for child in symbol.children]
            return symbol._concatenation_new_copy(new_children)  # noqa: SLF001

        # Return leaf
        return symbol
