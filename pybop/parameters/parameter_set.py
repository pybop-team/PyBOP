import json
import types
import pybamm
import pybop


class ParameterSet:
    """
    Handles the import and export of parameter sets for battery models.

    This class provides methods to load parameters from a JSON file and to export them
    back to a JSON file. It also includes custom logic to handle special cases, such
    as parameter values that require specific initialization.

    Parameters
    ----------
    json_path : str, optional
        Path to a JSON file containing parameter data. If provided, parameters will be imported from this file during initialization.
    params_dict : dict, optional
        A dictionary of parameters to initialize the ParameterSet with. If not provided, an empty dictionary is used.
    """

    def __init__(self, json_path=None, params_dict=None):
        self.json_path = json_path
        self.params = params_dict or {}
        self.chemistry = None

    def import_parameters(self, json_path=None):
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

        # Read JSON file
        if not self.params and self.json_path:
            with open(self.json_path, "r") as file:
                self.params = json.load(file)
                self._handle_special_cases()
        if self.params["chemistry"] is not None:
            self.chemistry = self.params["chemistry"]
        return self.params

    def _handle_special_cases(self):
        """
        Processes special cases for parameter values that require custom handling.

        For example, if the open-circuit voltage is specified as 'default', it will
        fetch the default value from the PyBaMM empirical Thevenin model.
        """
        if (
            "Open-circuit voltage [V]" in self.params
            and self.params["Open-circuit voltage [V]"] == "default"
        ):
            self.params[
                "Open-circuit voltage [V]"
            ] = pybop.empirical.Thevenin().default_parameter_values[
                "Open-circuit voltage [V]"
            ]

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
        if not self.params:
            raise ValueError("No parameters to export. Please import parameters first.")

        # Prepare a copy of the params to avoid modifying the original dict
        exportable_params = {**{"chemistry": self.chemistry}, **self.params.copy()}

        # Update parameter set
        if fit_params is not None:
            for i, param in enumerate(fit_params):
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
        return pybamm.ParameterValues(name).copy()
