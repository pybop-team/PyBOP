import warnings

import numpy as np
from pybamm import DummySolver, Parameter, ParameterValues, citations
from pybamm import lithium_ion as pybamm_lithium_ion
from pybamm import t as pybamm_t

from pybop import ParameterSet


class BaseWeppnerHuggins(pybamm_lithium_ion.BaseModel):
    """
    WeppnerHuggins Model for GITT. Credit: pybamm-param team.

    This model can be used with PyBOP through the `pybop.WeppnerHuggins` class.

    Parameters
    ----------
    name : str, optional
        The name of the model.
    **model_kwargs : optional
        Valid PyBaMM model option keys and their values, for example:
        parameter_set : pybamm.ParameterValues or dict, optional
            The parameters for the model. If None, default parameters provided by PyBaMM are used.
        options : dict, optional
            A dictionary of options to customise the behaviour of the PyBaMM model.
    """

    def __init__(self, name="Weppner & Huggins model", **model_kwargs):
        unused_keys = []
        for key in model_kwargs.keys():
            if key not in ["build", "parameter_set", "options"]:
                unused_keys.append(key)
        options = {"working electrode": "positive"}
        if model_kwargs.get("options", None) is not None:
            for key, value in model_kwargs["options"].items():
                if key in ["working electrode"]:
                    options[key] = value
                else:
                    unused_keys.append("options[" + key + "]")
        if any(unused_keys):
            unused_kwargs_warning = f"The input model_kwargs {unused_keys} are not currently used by the Weppner & Huggins model."
            warnings.warn(unused_kwargs_warning, UserWarning, stacklevel=2)

        super().__init__(options=options, name=name, build=True)
        self._summary_variables = []

        citations.register("""
            @article{Weppner1977,
            title={{Determination of the kinetic parameters
            of mixed-conducting electrodes and application to the system Li3Sb}},
            author={Weppner, W and Huggins, R A},
            journal={Journal of The Electrochemical Society},
            volume={124},
            number={10},
            pages={1569},
            year={1977},
            publisher={IOP Publishing}
            }
        """)

        ######################
        # Parameters
        ######################
        # Parameters are purely symbolic at this stage, and will be set by the
        # `ParameterValues` class when the model is processed.

        Q_th = Parameter("Theoretical electrode capacity [A.s]")

        U = Parameter("Reference voltage [V]")
        U_prime = Parameter("Derivative of the OCP wrt stoichiometry [V]")

        tau_d = Parameter("Particle diffusion time scale [s]")

        ######################
        # Input current (positive on discharge)
        ######################
        I = self.param.current_with_time

        ######################
        # Governing equations
        ######################
        # Surface stoichiometry
        sto_surf = 2 * I / (3 * Q_th) * (pybamm_t * tau_d / np.pi) ** 0.5
        # Linearised voltage
        V = U + U_prime * sto_surf

        ######################
        # (Some) variables
        ######################
        self.variables = {
            "Voltage [V]": V,
            "Time [s]": pybamm_t,
            "Current [A]": I,
            "Current variable [A]": I,  # for compatibility with pybamm.Experiment
        }

        # Set the built property on creation to prevent unnecessary model rebuilds
        self._built = True

    @property
    def default_geometry(self):
        return {}

    @property
    def default_parameter_values(self):
        parameter_dictionary = {
            "Current function [A]": 0.0024,
            "Reference voltage [V]": 0.0024,
            "Derivative of the OCP wrt stoichiometry [V]": -1,
            "Theoretical electrode capacity [A.s]": 15,
            "Particle diffusion time scale [s]": 3000,
        }
        return ParameterValues(values=parameter_dictionary)

    @property
    def default_submesh_types(self):
        return {}

    @property
    def default_var_pts(self):
        return {}

    @property
    def default_spatial_methods(self):
        return {}

    @property
    def default_solver(self):
        return DummySolver()

    @staticmethod
    def apply_parameter_grouping(parameter_set, electrode) -> dict:
        """
        A function to create an electrode parameter set from a standard
        PyBaMM parameter set.

        Parameters
        ----------
        parameter_set : Union[dict, pybop.ParameterSet, pybamm.ParameterValues]
            A dict-like object containing the parameter values.
        electrode : str
            Either "positive" or "negative" for the type of electrode.

        Returns
        -------
        dict
            A dictionary of the grouped parameters.
        """
        parameter_set = ParameterSet.to_pybamm(parameter_set)

        # Unpack physical parameters
        F = parameter_set["Faraday constant [C.mol-1]"]
        if electrode == "positive":
            alpha = parameter_set["Positive electrode active material volume fraction"]
            c_max = parameter_set[
                "Maximum concentration in positive electrode [mol.m-3]"
            ]
            L = parameter_set["Positive electrode thickness [m]"]
            R = parameter_set["Positive particle radius [m]"]
            D = parameter_set["Positive particle diffusivity [m2.s-1]"]
        elif electrode == "negative":
            alpha = parameter_set["Negative electrode active material volume fraction"]
            c_max = parameter_set[
                "Maximum concentration in negative electrode [mol.m-3]"
            ]
            L = parameter_set["Negative electrode thickness [m]"]
            R = parameter_set["Negative particle radius [m]"]
            D = parameter_set["Negative particle diffusivity [m2.s-1]"]
        else:
            raise ValueError(
                f"Unrecognised electrode type: {electrode}, "
                'expecting either "positive" or "negative".'
            )

        # Compute the cell area
        A = parameter_set["Electrode height [m]"] * parameter_set["Electrode width [m]"]

        # Grouped parameters
        Q_th = F * alpha * c_max * L * A
        tau_d = R**2 / D

        return {
            "Current function [A]": parameter_set["Current function [A]"],
            "Reference voltage [V]": 4,
            "Derivative of the OCP wrt stoichiometry [V]": -1,
            "Theoretical electrode capacity [A.s]": Q_th,
            "Particle diffusion time scale [s]": tau_d,
        }
