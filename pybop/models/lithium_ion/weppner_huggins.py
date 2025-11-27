import numpy as np
from pybamm import DummySolver, Parameter, ParameterValues, citations, lithium_ion
from pybamm import t as pybamm_t


class WeppnerHuggins(lithium_ion.BaseModel):
    """
    Represents the Weppner & Huggins model to fit diffusion coefficients to GITT data.

    Note: the working electrode is the positive electrode.

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
    def default_parameter_values(self) -> ParameterValues:
        param = ParameterValues("Xu2019")
        return self.create_grouped_parameters(param)

    @property
    def default_quick_plot_variables(self):
        return [
            "Current [A]",
            "Voltage [V]",
        ]

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
    def create_grouped_parameters(parameter_values: ParameterValues) -> ParameterValues:
        """
        Create a parameter set for the Weppner & Huggins model from a
        PyBaMM lithium-ion ParameterValues object.

        Note: the working electrode is the positive electrode.

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
        alpha = param["Positive electrode active material volume fraction"]
        c_max = param["Maximum concentration in positive electrode [mol.m-3]"]
        L = param["Positive electrode thickness [m]"]
        R = param["Positive particle radius [m]"]
        D = param["Positive particle diffusivity [m2.s-1]"]

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
        parameter_values = ParameterValues(values=parameter_dictionary)
        parameter_values._set_initial_state = set_initial_state  # noqa: SLF001
        return parameter_values


def set_initial_state(
    initial_value,
    parameter_values,
    direction=None,
    param=None,
    inplace=True,
    options=None,
    inputs=None,
    tol=1e-6,
):
    """
    Set the value of the initial state of charge.

    Parameters
    ----------
    initial_value : float
        Target initial value.
        If float, interpreted as SOC, must be between 0 and 1.
        If string e.g. "4 V", interpreted as voltage, must be between V_min and V_max.
    parameter_values : :class:`pybamm.ParameterValues`
        Parameters and their corresponding values.
    param : :class:`pybamm.LithiumIonParameters`, optional
        The symbolic parameter set to use for the simulation.
        If not provided, the default parameter set will be used.
    inplace: bool, optional
        If True, replace the parameters values in place. Otherwise, return a new set of
        parameter values. Default is True.
    options : dict-like, optional
        A dictionary of options to be passed to the model, see
        :class:`pybamm.BatteryModelOptions`.
    inputs : dict, optional
        A dictionary of input parameters to pass to the model when solving.
    tol : float, optional
        The tolerance for the solver used to compute the initial stoichiometries.
        A lower value results in higher precision but may increase computation time.
        Default is 1e-6.
    """
    parameter_values = parameter_values if inplace else parameter_values.copy()

    if isinstance(initial_value, int | float):
        if not 0 <= initial_value <= 1:
            raise ValueError("Initial SOC should be between 0 and 1")
        parameter_values["Initial SoC"] = initial_value

    else:
        raise ValueError("Initial value must be a float between 0 and 1.")

    return parameter_values
