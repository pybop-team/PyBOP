import numpy as np
from pybamm import DummySolver, Parameter, ParameterValues, citations, lithium_ion
from pybamm import t as pybamm_t


class WeppnerHuggins(lithium_ion.BaseModel):
    """
    Represents the Weppner & Huggins model to fit diffusion coefficients to GITT data.

    Parameters
    ----------
    name : str, optional
        The name of the model.
    **model_kwargs : optional
        Valid PyBaMM model option keys and their values, for example:
        options : dict, optional
            A dictionary of options to customise the behaviour of the PyBaMM model.
        build : bool, optional
            If True, the model is built upon creation (default: False).
    """

    def __init__(self, name="Weppner & Huggins model", **model_kwargs):
        super().__init__(name=name, **model_kwargs)
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
    def create_grouped_parameters(
        parameter_values: ParameterValues, electrode: str
    ) -> ParameterValues:
        """
        Create a parameter set for the Weppner & Huggins model from a
        PyBaMM lithium-ion ParameterValues object.

        Parameters
        ----------
        parameter_values : pybamm.ParameterValues
            Parameters and their corresponding values.

        Returns
        -------
        parameter_values : pybamm.ParameterValues
            A new set of parameters and their values.
        """
        param = parameter_values

        # Unpack physical parameters
        F = param["Faraday constant [C.mol-1]"]
        if electrode == "positive":
            alpha = param["Positive electrode active material volume fraction"]
            c_max = param["Maximum concentration in positive electrode [mol.m-3]"]
            L = param["Positive electrode thickness [m]"]
            R = param["Positive particle radius [m]"]
            D = param["Positive particle diffusivity [m2.s-1]"]
        elif electrode == "negative":
            alpha = param["Negative electrode active material volume fraction"]
            c_max = param["Maximum concentration in negative electrode [mol.m-3]"]
            L = param["Negative electrode thickness [m]"]
            R = param["Negative particle radius [m]"]
            D = param["Negative particle diffusivity [m2.s-1]"]
        else:
            raise ValueError(
                f"Unrecognised electrode type: {electrode}, "
                'expecting either "positive" or "negative".'
            )

        # Compute the cell area
        A = param["Electrode height [m]"] * param["Electrode width [m]"]

        # Grouped parameters
        Q_th = F * alpha * c_max * L * A
        tau_d = R**2 / D

        parameter_dictionary = {
            "Current function [A]": param["Current function [A]"],
            "Reference voltage [V]": 4,
            "Derivative of the OCP wrt stoichiometry [V]": -1,
            "Theoretical electrode capacity [A.s]": Q_th,
            "Particle diffusion time scale [s]": tau_d,
        }
        return ParameterValues(values=parameter_dictionary)
