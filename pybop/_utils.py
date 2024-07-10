import numpy as np
import pybamm


def is_numeric(x):
    """
    Check if a variable is numeric.
    """
    return isinstance(x, (int, float, np.number))


class SymbolReplacer:
    """
    Helper class to replace all instances of one or more symbols in an expression tree
    with another symbol, as defined by the dictionary `symbol_replacement_map`

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
        symbol_replacement_map,
        processed_symbols=None,
        process_initial_conditions=True,
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

        new_events = []
        for event in unprocessed_model.events:
            pybamm.logger.verbose(f"Replacing symbols in event'{event.name}''")
            new_events.append(
                pybamm.Event(
                    event.name, self.process_symbol(event.expression), event.event_type
                )
            )
        model.events = new_events

        pybamm.logger.info(f"Finish replacing symbols in {model.name}")

        return model

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

        try:
            return self._processed_symbols[symbol]
        except KeyError:
            self._processed_symbols[symbol] = self._process_symbol(symbol)
            return self._processed_symbols[symbol]

    def _process_symbol(self, symbol):
        """See :meth:`Simplification.process_symbol()`."""
        if symbol in self._symbol_replacement_map.keys():
            return self._symbol_replacement_map[symbol]

        elif isinstance(symbol, pybamm.BinaryOperator):
            left, right = symbol.children
            # process children
            new_left = self.process_symbol(left)  # BP: We could do this recursively
            new_right = self.process_symbol(right)
            # Return a new copy with the replaced symbols
            return symbol._binary_new_copy(new_left, new_right)

        elif isinstance(symbol, pybamm.UnaryOperator):
            new_child = self.process_symbol(symbol.child)
            # Return a new copy with the replaced symbols
            return symbol._unary_new_copy(new_child)

        elif isinstance(symbol, pybamm.Function):
            new_children = [self.process_symbol(child) for child in symbol.children]
            # Return a new copy with the replaced symbols
            return symbol._function_new_copy(new_children)

        elif isinstance(symbol, pybamm.Concatenation):
            new_children = [self.process_symbol(child) for child in symbol.children]
            # Return a new copy with the replaced symbols
            return symbol._concatenation_new_copy(new_children)

        else:
            # Only other option is that the symbol is a leaf (doesn't have children)
            # In this case, since we have already ruled out that the symbol is one of
            # the symbols that needs to be replaced, we can just return the symbol
            return symbol
