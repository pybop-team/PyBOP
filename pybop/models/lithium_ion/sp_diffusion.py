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


class SPDiffusion(pybamm_lithium_ion.BaseModel):
    """
    Diffusion model for a single, spherical particle representing a half-cell for GITT.

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

    def __init__(self, name="Single Particle Diffusion Model", **model_kwargs):
        super().__init__(name=name, **model_kwargs)

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
        j = -I / (3 * Q_th)
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
    def default_parameter_values(self) -> ParameterValues:
        param = ParameterValues("Xu2019")
        return self.create_grouped_parameters(param)

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
    def create_grouped_parameters(parameter_values: ParameterValues) -> ParameterValues:
        """
        Create a parameter set for the Single Particle Diffusion Model from a
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
        sto_init = (
            param["Initial concentration in positive electrode [mol.m-3]"] / c_max
        )
        ocp = param["Positive electrode OCP [V]"]

        # Compute the cell area
        A = param["Electrode height [m]"] * param["Electrode width [m]"]

        # Grouped parameters
        Q_th = F * alpha * c_max * L * A
        tau_d = R**2 / D

        parameter_dictionary = {
            "Nominal cell capacity [A.h]": param["Nominal cell capacity [A.h]"],
            "Current function [A]": param["Current function [A]"],
            "Initial stoichiometry": sto_init,
            "Electrode OCP [V]": ocp,
            "Theoretical electrode capacity [A.s]": Q_th,
            "Particle diffusion time scale [s]": tau_d,
            "Series resistance [Ohm]": 1,
        }
        return ParameterValues(values=parameter_dictionary)
