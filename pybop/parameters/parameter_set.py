import copy
import json
import types
from numbers import Number
from typing import Optional, Union

import numpy as np
from pybamm import (
    FunctionParameter,
    LithiumIonParameters,
    Parameter,
    ParameterValues,
    Scalar,
    Symbol,
    parameter_sets,
)


class ParameterSet:
    """
    Handles the import and export of parameter sets for battery models.

    This class provides methods to load parameters from a JSON file and to export them
    back to a JSON file. It also includes custom logic to handle special cases, such
    as parameter values that require specific initialization.

    Parameters
    ----------
    parameter_set : Union[str, dict, ParameterValues], optional
        A dictionary of parameters to initialise the ParameterSet with. If not provided, an empty dictionary is used.
    json_path : str, optional
        Path to a JSON file containing parameter data. If provided, parameters will be imported from this file during
        initialisation.
    formation_concentrations : bool, optional
            If True, re-calculates the initial concentrations of lithium in the active material (default: False).
    """

    def __init__(
        self,
        parameter_set: Union[str, dict, ParameterValues] = None,
        json_path: Optional[str] = None,
        formation_concentrations: Optional[bool] = False,
    ):
        if parameter_set is not None and json_path is not None:
            raise ValueError(
                "ParameterSet needs either a parameter_set or json_path as an input, not both."
            )
        self._json_path = None
        self.parameter_values = None
        self.chemistry = None
        self.formation_concentrations = formation_concentrations

        if json_path is not None:
            self.import_parameters(json_path)
        else:
            self.parameter_values = self.to_pybamm(parameter_set)

        if self.parameter_values is not None:
            self.chemistry = self.parameter_values.get("chemistry", None)

            if self.formation_concentrations:
                set_formation_concentrations(self.parameter_values)

    def __setitem__(self, key, value):
        self.parameter_values[key] = value

    def __getitem__(self, key):
        return self.parameter_values[key]

    @staticmethod
    def evaluate_symbol(symbol: Union[Symbol, Number], params: dict):
        """
        Evaluate a parameter in the parameter set.

        Parameters
        ----------
        symbol : pybamm.Symbol or Number
            The parameter to evaluate.

        Returns
        -------
        float
            The value of the parameter.
        """
        if isinstance(symbol, (Number, np.float64)):
            return symbol
        if isinstance(symbol, Scalar):
            return symbol.value
        if isinstance(symbol, (Parameter, FunctionParameter)):
            return ParameterSet.evaluate_symbol(params[symbol.name], params)
        new_children = [
            Scalar(ParameterSet.evaluate_symbol(child, params))
            for child in symbol.children
        ]
        return symbol.create_copy(new_children).evaluate()

    def keys(self) -> list:
        """
        A list of parameter names
        """
        return list(self.parameter_values.keys())

    def update(self, params_dict: dict = None, check_already_exists: bool = True):
        """
        Update the parameter dictionary.

        Parameters
        ----------
        params_dict : dict
            A dictionary of parameters and values used to update the parameter values
        check_already_exists : bool, optional
            Whether to check that a parameter in `params_dict` already exists when trying
            to update it. This is to avoid cases where an intended change in the parameters
            is ignored due a typo in the parameter name (default: True).
        """
        self.parameter_values.update(
            params_dict, check_already_exists=check_already_exists
        )

    def import_parameters(self, json_path: Optional[str] = None):
        """
        Imports parameters from a JSON file specified by the `json_path` attribute.

        If a `json_path` is provided at initialization or as an argument, that JSON file
        is loaded and the parameters are stored in the `params` attribute. Special cases
        are handled appropriately.

        Parameters
        ----------
        json_path : str, optional
            Path to the JSON file from which to import parameters. If provided, it overrides the instance's `json_path`.

        Returns
        -------
        dict
            The dictionary containing the imported parameters.

        Raises
        ------
        FileNotFoundError
            If the specified JSON file cannot be found.
        """
        if self.parameter_values:
            raise ValueError("Parameter set already constructed.")

        # Read JSON file
        if json_path:
            self._json_path = json_path
            try:
                self.parameter_values = ParameterValues.create_from_bpx(self._json_path)
            except Exception:
                print(
                    "The JSON file was not recognised as a BPX parameter set. Importing as a JSON file."
                )
                with open(self._json_path) as file:
                    params = json.load(file)
                    self.parameter_values = ParameterValues(params)
        else:
            raise ValueError("No path was provided.")

        self.chemistry = self.parameter_values.get("chemistry", None)

        if self.formation_concentrations:
            set_formation_concentrations(self.parameter_values)

        return self.parameter_values

    def export_parameters(self, output_json_path, fit_params=None):
        """
        Exports parameters to a JSON file specified by `output_json_path`.

        The current state of the `params` attribute is written to the file. If `fit_params`
        is provided, these parameters are updated before export. Non-serializable values
        are handled and noted in the output JSON.

        Parameters
        ----------
        output_json_path : str
            The file path where the JSON output will be saved.
        fit_params : list of fitted parameter objects, optional
            Parameters that have been fitted and need to be included in the export.

        Raises
        ------
        ValueError
            If there are no parameters to export.
        """
        if not self.parameter_values:
            raise ValueError("No parameters to export. Please import parameters first.")

        # Prepare a copy of the params to avoid modifying the original dict
        exportable_params = {
            **{"chemistry": self.chemistry},
            **self.parameter_values.copy(),
        }

        # Update parameter set
        if fit_params is not None:
            for _i, param in enumerate(fit_params):
                exportable_params.update({param.name: param.value})

        # Replace non-serializable values
        for key, value in exportable_params.items():
            if isinstance(value, types.FunctionType) or not self.is_json_serializable(
                value
            ):
                exportable_params[key] = "Unable to write value to JSON file"

        # Write parameters to JSON file
        with open(output_json_path, "w") as file:
            json.dump(exportable_params, file, indent=4)

    def is_json_serializable(self, value):
        """
        Determines if the given `value` can be serialized to JSON format.

        Parameters
        ----------
        value : any
            The value to check for JSON serializability.

        Returns
        -------
        bool
            True if the value is JSON serializable, False otherwise.
        """
        try:
            json.dumps(value)
            return True
        except (TypeError, OverflowError):
            return False

    def copy(self):
        new_copy = ParameterSet(
            parameter_set=self.parameter_values.copy(),
            json_path=copy.copy(self._json_path),
            formation_concentrations=copy.copy(self.formation_concentrations),
        )
        return new_copy

    @classmethod
    def pybamm(cls, name):
        """
        Retrieves a PyBaMM parameter set by name.

        Parameters
        ----------
        name : str
            The name of the PyBaMM parameter set to retrieve.

        Returns
        -------
        pybamm.ParameterValues
            A PyBaMM parameter set corresponding to the provided name.
        """

        msg = f"Parameter set '{name}' is not a valid PyBaMM parameter set. Available parameter sets are: {list(parameter_sets)}"

        if name not in list(parameter_sets):
            raise ValueError(msg)

        return ParameterValues(name).copy()

    @classmethod
    def to_pybamm(cls, parameter_set):
        """
        Converts a parameter set to a new PyBaMM ParameterValues object.
        """
        if parameter_set is None:
            return None
        elif isinstance(parameter_set, str):
            # Use class method
            return cls.pybamm(parameter_set)
        elif isinstance(parameter_set, dict):
            return ParameterValues(parameter_set)
        elif isinstance(parameter_set, ParameterValues):
            return parameter_set.copy()
        else:
            return parameter_set.parameter_values.copy()


def set_formation_concentrations(parameter_set):
    """
    Compute the concentration of lithium in the positive electrode assuming that
    all lithium in the active material originated from the positive electrode.

    Only perform the calculation if an initial concentration exists for both
    electrodes, i.e. it is not a half cell.

    Parameters
    ----------
    parameter_set : pybamm.ParameterValues
        A PyBaMM parameter set containing standard lithium ion parameters.
    """
    if (
        all(
            key in parameter_set.keys()
            for key in [
                "Initial concentration in negative electrode [mol.m-3]",
                "Initial concentration in positive electrode [mol.m-3]",
            ]
        )
        and parameter_set["Initial concentration in negative electrode [mol.m-3]"] > 0
    ):
        # Obtain the total amount of lithium in the active material
        Q_Li_particles_init = parameter_set.evaluate(
            LithiumIonParameters().Q_Li_particles_init
        )

        # Convert this total amount to a concentration in the positive electrode
        c_init = (
            Q_Li_particles_init
            * 3600
            / (
                parameter_set["Positive electrode active material volume fraction"]
                * parameter_set["Positive electrode thickness [m]"]
                * parameter_set["Electrode height [m]"]
                * parameter_set["Electrode width [m]"]
                * parameter_set["Faraday constant [C.mol-1]"]
            )
        )

        # Update the initial lithium concentrations
        parameter_set.update(
            {"Initial concentration in negative electrode [mol.m-3]": 0}
        )
        parameter_set.update(
            {"Initial concentration in positive electrode [mol.m-3]": c_init}
        )
