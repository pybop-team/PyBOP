import warnings

import pybamm
from pybamm import (
    Event,
    FunctionParameter,
    Parameter,
    ParameterValues,
    PrimaryBroadcast,
    Scalar,
    SpatialVariable,
    Variable,
)
from pybamm import lithium_ion as pybamm_lithium_ion
from pybamm import t as pybamm_t
from pybamm.input.parameters.lithium_ion.Xu2019 import (
    nmc_ocp_Xu2019,
)

from pybop import ParameterSet


class BaseSPDiffusion(pybamm_lithium_ion.BaseModel):
    """
    Diffusion model for a single, spherical particle representing a half-cell for GITT.

    This model can be used with PyBOP through the `pybop.SPDiffusion` class.

    Parameters
    ----------
    name : str, optional
        The name of the model.
    electrode : str, optional
        Either "positive" or "negative" depending on the type of electrode.
    **model_kwargs : optional
        Valid PyBaMM model option keys and their values, for example:
        parameter_set : pybamm.ParameterValues or dict, optional
            The parameters for the model. If None, default parameters provided by PyBaMM are used.
        geometry : dict, optional
            The geometry definitions for the model. If None, default geometry from PyBaMM is used.
        submesh_types : dict, optional
            The types of submeshes to use. If None, default submesh types from PyBaMM are used.
        var_pts : dict, optional
            The discretization points for each variable in the model. If None, default points from PyBaMM are used.
        spatial_methods : dict, optional
            The spatial methods used for discretization. If None, default spatial methods from PyBaMM are used.
        solver : pybamm.Solver, optional
            The solver to use for simulating the model. If None, the default solver from PyBaMM is used.
        build : bool, optional
            If True, the model is built upon creation (default: False).
        options : dict, optional
            A dictionary of options to customise the behaviour of the PyBaMM model.
    """

    def __init__(
        self,
        name="Single Particle Diffusion Model",
        electrode="negative",
        **model_kwargs,
    ):
        unused_keys = []
        for key in model_kwargs.keys():
            if key not in ["build", "parameter_set", "options"]:
                unused_keys.append(key)
        if model_kwargs.get("build", True) is False:
            unused_keys.append("build")
        if model_kwargs.get("options", None) is not None:
            for key in model_kwargs["options"].keys():
                unused_keys.append("options[" + key + "]")
        if any(unused_keys):
            unused_kwargs_warning = f"The input model_kwargs {unused_keys} are not currently used by the SP Diffusion Model."
            warnings.warn(unused_kwargs_warning, UserWarning, stacklevel=2)

        super().__init__(options={}, name=name, build=True)

        pybamm.citations.register("Xu2019")  # for the OCP

        ######################
        # Variables
        ######################
        # Variables that depend on time only are created without a domain
        Q = Variable("Discharge capacity [A.h]")
        Qt = Variable("Throughput capacity [A.h]")

        # Variables that vary spatially are created with a domain
        sto = Variable("Particle stoichiometry", domain="particle")
        sto_surf = pybamm.surf(sto)

        # Events specify points at which a solution should terminate
        self.events += [
            Event(
                "Minimum particle surface stoichiometry",
                pybamm.min(sto_surf) - 0.01,
            ),
            Event(
                "Maximum particle surface stoichiometry",
                (1 - 0.01) - pybamm.max(sto_surf),
            ),
        ]

        ######################
        # Parameters
        ######################
        # Parameters are purely symbolic at this stage, and will be set by the
        # `ParameterValues` class when the model is processed.

        Q_th = Parameter("Theoretical electrode capacity [A.s]")

        sto_init = Parameter("Initial stoichiometry")

        ######################
        # Input current (positive on discharge)
        ######################
        I = self.param.current_with_time

        ######################
        # State of Charge
        ######################
        # The `rhs` dictionary contains differential equations, with the key being the
        # variable in the d/dt
        self.rhs[Q] = I / 3600
        self.rhs[Qt] = abs(I) / 3600
        # Initial conditions must be provided for the ODEs
        self.initial_conditions[Q] = Scalar(0)
        self.initial_conditions[Qt] = Scalar(0)

        ######################
        # Diffusion within the particle
        ######################
        # The div and grad operators will be converted to the appropriate matrix
        # multiplication at the discretisation stage
        self.rhs[sto] = pybamm.div(pybamm.grad(sto) / self.tau_d(sto))

        # Boundary conditions must be provided for equations with spatial derivatives
        if electrode == "positive":
            j = -I / (3 * Q_th)
        else:
            j = I / (3 * Q_th)
        self.boundary_conditions[sto] = {
            "left": (Scalar(0), "Neumann"),
            "right": (-self.tau_d(sto_surf) * j, "Neumann"),
        }

        self.initial_conditions[sto] = sto_init

        ######################
        # Cell voltage
        ######################
        U = self.U(sto_surf)
        V = U - self.R0(sto_surf) * I

        # Save the initial OCV
        self.param.ocv_init = self.U(sto_init)

        ######################
        # (Some) variables
        ######################
        self.variables = {
            "Particle stoichiometry": sto,
            "Particle surface stoichiometry": PrimaryBroadcast(sto_surf, "particle"),
            "Time [s]": pybamm_t,
            "Current [A]": I,
            "Current variable [A]": I,  # for compatibility with pybamm.Experiment
            "Discharge capacity [A.h]": Q,
            "Throughput capacity [A.h]": Qt,
            "Voltage [V]": V,
            "Open-circuit voltage [V]": U,
        }

    def U(self, sto):
        """
        Dimensional open-circuit potential [V], calculated as U(x) = U_ref(x).
        Credit: PyBaMM
        """
        # bound stoichiometry between tol and 1-tol. Adding 1/sto + 1/(sto-1) later
        # will ensure that ocp goes to +- infinity if sto goes into that region
        # anyway
        tol = pybamm.settings.tolerances["U__c_s"]
        sto = pybamm.maximum(pybamm.minimum(sto, 1 - tol), tol)
        inputs = {"Particle surface stoichiometry": sto}
        u_ref = FunctionParameter("Electrode OCP [V]", inputs)

        # add a term to ensure that the OCP goes to infinity at 0 and -infinity at 1
        # this will not affect the OCP for most values of sto
        out = u_ref + 1e-6 * (1 / sto + 1 / (sto - 1))

        out.print_name = r"U(c^\mathrm{surf}_\mathrm{s})"
        return out

    def tau_d(self, sto):
        """
        Diffusion time scale [s] for lithium in the particles.
        """
        inputs = {"Particle surface stoichiometry": sto}
        return FunctionParameter("Particle diffusion time scale [s]", inputs)

    def R0(self, sto):
        """
        Series resistance [Ohm].
        """
        inputs = {"Particle surface stoichiometry": sto}
        return FunctionParameter("Series resistance [Ohm]", inputs)

    def build_model(self):
        """
        Build model variables and equations
        Credit: PyBaMM
        """
        self._build_model()

        self._built = True
        pybamm.logger.info(f"Finish building {self.name}")

    @property
    def default_parameter_values(self):
        parameter_dictionary = {
            "Nominal cell capacity [A.h]": 0.0024,
            "Current function [A]": 0.0024,
            "Initial stoichiometry": 0.9,
            "Electrode OCP [V]": nmc_ocp_Xu2019,
            "Theoretical electrode capacity [A.s]": 15,
            "Particle diffusion time scale [s]": 3000,
            "Series resistance [Ohm]": 1,
        }
        return ParameterValues(values=parameter_dictionary)

    @property
    def default_quick_plot_variables(self):
        return [
            "Particle stoichiometry",
            "Particle surface stoichiometry",
            "Current [A]",
            {"Open-circuit voltage [V]", "Voltage [V]"},
        ]

    @property
    def default_var_pts(self):
        r = SpatialVariable("r", domain=["particle"], coord_sys="spherical polar")
        return {r: 20}

    @property
    def default_geometry(self):
        r = SpatialVariable("r", domain=["particle"], coord_sys="spherical polar")
        return {"particle": {r: {"min": 0, "max": 1}}}

    @property
    def default_submesh_types(self):
        return {"particle": pybamm.Uniform1DSubMesh}

    @property
    def default_spatial_methods(self):
        return {"particle": pybamm.FiniteVolume()}

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
            sto_init = (
                parameter_set["Initial concentration in positive electrode [mol.m-3]"]
                / c_max
            )
            ocp = parameter_set["Positive electrode OCP [V]"]
        elif electrode == "negative":
            alpha = parameter_set["Negative electrode active material volume fraction"]
            c_max = parameter_set[
                "Maximum concentration in negative electrode [mol.m-3]"
            ]
            L = parameter_set["Negative electrode thickness [m]"]
            R = parameter_set["Negative particle radius [m]"]
            D = parameter_set["Negative particle diffusivity [m2.s-1]"]
            sto_init = (
                parameter_set["Initial concentration in negative electrode [mol.m-3]"]
                / c_max
            )
            ocp = parameter_set["Negative electrode OCP [V]"]
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
            "Nominal cell capacity [A.h]": parameter_set["Nominal cell capacity [A.h]"],
            "Current function [A]": parameter_set["Current function [A]"],
            "Initial stoichiometry": sto_init,
            "Electrode OCP [V]": ocp,
            "Theoretical electrode capacity [A.s]": Q_th,
            "Particle diffusion time scale [s]": tau_d,
            "Series resistance [Ohm]": 1,
        }
