import re
from dataclasses import dataclass, field

import numpy as np
import pybamm


@dataclass(frozen=True)
class FailedVariable:
    """
    Container for a failed PyBaMM variable that returns np.inf.

    Args:
        name: Variable name
        data: Array data, defaults to [np.inf]
        sensitivities: Sensitivity data mapping parameter names to arrays
    """

    name: str
    data: np.ndarray = field(default_factory=lambda: np.asarray([np.inf]))
    sensitivities: {str, np.ndarray} = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate inputs after initialisation."""
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Variable name must be a non-empty string")

        if not isinstance(self.data, np.ndarray):
            object.__setattr__(self, "data", np.asarray(self.data))


class FailedSolution:
    """
    Container for a failed PyBaMM solution that returns np.inf for processed variables.

    This class mimics the interface of a successful PyBaMM solution but returns
    infinity values to indicate failure while maintaining API compatibility.

    Args:
        variable_names: List of variable names in the solution
        parameter_names: List of parameter names for sensitivity analysis

    Example:
        >>> solution = FailedSolution(["Voltage [V]"], ["Negative particle radius [m]"])
        >>> voltage = solution["Voltage [V]"]
        >>> print(voltage.data)  # np.ndarray([inf])
    """

    def __init__(self, variable_names: list[str], parameter_names: list[str]):
        self._validate_inputs(variable_names, parameter_names)
        self._variable_names = variable_names
        self._parameter_names = parameter_names

        # Solution metadata
        self.cycles: int | None = None
        self.termination: str = "failure"
        self.solve_time: float = 0.0
        self.integration_time: float = 0.0
        self._t_eval: np.ndarray = np.asarray([0.0])

        # Initialise failed variables
        self._variables: {str, FailedVariable} = pybamm.FuzzyDict()
        self._initialise_variables()

    def _validate_inputs(
        self, variable_names: list[str], parameter_names: list[str] | None
    ) -> None:
        """Validate constructor inputs."""
        if not variable_names:
            raise ValueError("variable_names cannot be empty")

        if not all(isinstance(name, str) and name.strip() for name in variable_names):
            raise ValueError("All variable names must be non-empty strings")

        if parameter_names is not None:
            if not all(
                isinstance(name, str) and name.strip() for name in parameter_names
            ):
                raise ValueError("All parameter names must be non-empty strings")

    def _initialise_variables(self) -> None:
        """Initialise all variables with failed state."""
        inf_array = np.asarray([np.inf])

        for var_name in self._variable_names:
            sensitivities = (
                {"all": [inf_array.copy() for _ in self._parameter_names]}
                if self._parameter_names
                else {}
            )

            self._variables[var_name] = FailedVariable(
                name=var_name, data=inf_array.copy(), sensitivities=sensitivities
            )

    def __getattr__(self, name):
        # Return self for any method calls to allow chaining
        return self

    def __getitem__(self, key):
        return self._variables[key]

    def plot(self, *args, **kwargs):
        print("Cannot plot a failed solution")
        return None

    def save(self, *args, **kwargs):
        print("Cannot save a failed solution")
        return None

    def copy(self):
        return FailedSolution(self._variable_names, self._parameter_names)

    @property
    def t_eval(self) -> np.ndarray:
        """Time evaluation points (returns [inf] for failed solutions)."""
        return self._t_eval

    @property
    def variable_names(self) -> list[str]:
        """Get list of variable names (read-only)."""
        return self._variable_names.copy()

    @property
    def parameter_names(self) -> list[str]:
        """Get list of parameter names (read-only)."""
        return self._parameter_names.copy()

    def keys(self) -> list[str]:
        """Get all variable names."""
        return list(self._variables.keys())

    def values(self) -> list[FailedVariable]:
        """Get all variables."""
        return list(self._variables.values())

    def items(self) -> list[tuple[str, FailedVariable]]:
        """Get all variable name-value pairs."""
        return list(self._variables.items())


def add_spaces(string):
    """
    Return the class name as a string with spaces before each new capitalised word.
    """
    re_outer = re.compile(r"([^A-Z ])([A-Z])")
    re_inner = re.compile(r"(?<!^)([A-Z])([^A-Z])")
    return re_outer.sub(r"\1 \2", re_inner.sub(r" \1\2", string))


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
            new model with parameter values set. Default is True.
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
