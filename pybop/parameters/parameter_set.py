import json
import types

from pybamm import LithiumIonParameters, ParameterValues, parameter_sets


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

    def __call__(self):
        return self.params

    def __setitem__(self, key, value):
        self.params[key] = value

    def __getitem__(self, key):
        return self.params[key]

    def keys(self) -> list:
        """
        A list of parameter names
        """
        return list(self.params.keys())

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
            with open(self.json_path) as file:
                self.params = json.load(file)
        else:
            raise ValueError(
                "Parameter set already constructed, or path to json file not provided."
            )
        if self.params["chemistry"] is not None:
            self.chemistry = self.params["chemistry"]
        return self.params

    def import_from_bpx(self, json_path=None):
        """
        Imports parameters from a JSON file in the BPX format specified by the `json_path`
        attribute.
        Credit: PyBaMM

        If a `json_path` is provided at initialization or as an argument, that JSON file
        is loaded and the parameters are stored in the `params` attribute.

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
            self.params = ParameterValues.create_from_bpx(self.json_path)
        else:
            raise ValueError(
                "Parameter set already constructed, or path to bpx file not provided."
            )
        return self.params

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

    @classmethod
    def pybamm(cls, name, formation_concentrations=False):
        """
        Retrieves a PyBaMM parameter set by name.

        Parameters
        ----------
        name : str
            The name of the PyBaMM parameter set to retrieve.
        set_formation_concentrations : bool, optional
            If True, re-calculates the initial concentrations of lithium in the active material (default: False).

        Returns
        -------
        pybamm.ParameterValues
            A PyBaMM parameter set corresponding to the provided name.
        """

        msg = f"Parameter set '{name}' is not a valid PyBaMM parameter set. Available parameter sets are: {list(parameter_sets)}"

        if name not in list(parameter_sets):
            raise ValueError(msg)

        parameter_set = ParameterValues(name).copy()

        if formation_concentrations:
            set_formation_concentrations(parameter_set)

        return parameter_set


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
