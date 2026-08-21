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
from pybamm import t as pybamm_t

from pybop.models.li_half_cell.base_model import BaseHalfCellModel


class SPDiffusion(BaseHalfCellModel):
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
        sto_p = Variable("Positive particle stoichiometry", domain="positive particle")

        # Surf takes the surface value of a variable, i.e. its boundary value on the
        # right side. This is also accessible via `boundary_value(x, "right")`, with
        # "left" providing the boundary value of the left side
        sto_p_surf = pybamm.surf(sto_p)

        # Events specify points at which a solution should terminate
        tol = pybamm.settings.tolerances["U__c_s"]
        self.events += [
            Event(
                "Minimum positive particle surface stoichiometry",
                pybamm.min(sto_p_surf) - tol,
            ),
            Event(
                "Maximum positive particle surface stoichiometry",
                (1 - tol) - pybamm.max(sto_p_surf),
            ),
        ]

        ######################
        # Parameters
        ######################
        # Parameters are purely symbolic at this stage, and will be set by the
        # `ParameterValues` class when the model is processed.

        # Grouped parameters
        Q_th_p = Parameter("Theoretical electrode capacity [A.h]") * 3600

        sto_p_init = Parameter("Initial stoichiometry")

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
        self.rhs[sto_p] = pybamm.div(pybamm.grad(sto_p) / self.tau_d(sto_p))

        # Boundary conditions must be provided for equations with spatial derivatives
        j_p = -I / (3 * Q_th_p)
        self.boundary_conditions[sto_p] = {
            "left": (Scalar(0), "Neumann"),
            "right": (-self.tau_d(sto_p_surf) * j_p, "Neumann"),
        }

        self.initial_conditions[sto_p] = sto_p_init

        ######################
        # Cell voltage
        ######################
        U = self.U(sto_p_surf)
        V = U - self.R0(sto_p_surf) * I

        # Save the initial OCV
        self.param.ocv_init = self.U(sto_p_init)

        # Events specify points at which a solution should terminate
        self.events += [
            Event("Minimum voltage [V]", V - self.param.voltage_low_cut),
            Event("Maximum voltage [V]", self.param.voltage_high_cut - V),
        ]

        ######################
        # (Some) variables
        ######################
        self.variables = {
            "Positive particle stoichiometry": sto_p,
            "Positive particle surface stoichiometry": PrimaryBroadcast(
                sto_p_surf, "positive particle"
            ),
            "Time [s]": pybamm_t,
            "Current [A]": I,
            "Current variable [A]": I,  # for compatibility with pybamm.Experiment
            "Discharge capacity [A.h]": Q,
            "Throughput capacity [A.h]": Qt,
            "Voltage [V]": V,
            "Voltage expression [V]": V,  # for compatibility with "voltage as a state"
            "Open-circuit voltage [V]": U,
        }

    def U(self, sto):
        """
        Dimensional open-circuit potential [V].
        Credit: PyBaMM
        """
        inputs = {"Positive particle surface stoichiometry": sto}
        out = FunctionParameter("Positive electrode OCP [V]", inputs)

        out.print_name = r"U_\mathrm{p}(c^\mathrm{surf}_\mathrm{s,p})"
        return out

    def tau_d(self, sto):
        """
        Dimensional solid-state diffusion time scale [s].
        """
        inputs = {"Positive particle surface stoichiometry": sto}
        return FunctionParameter("Positive particle diffusion time scale [s]", inputs)

    def R0(self, sto):
        """
        Series resistance [Ohm].
        """
        inputs = {"Positive particle surface stoichiometry": sto}
        return FunctionParameter("Series resistance [Ohm]", inputs)

    @property
    def default_parameter_values(self) -> ParameterValues:
        param = ParameterValues("Xu2019")
        return self.create_grouped_parameters(param)

    @property
    def default_quick_plot_variables(self):
        return [
            "Positive particle stoichiometry",
            "Positive particle surface stoichiometry",
            "Current [A]",
            {"Open-circuit voltage [V]", "Voltage [V]"},
        ]

    @property
    def default_var_pts(self):
        r_p = SpatialVariable(
            "r_p", domain=["positive particle"], coord_sys="spherical polar"
        )
        return {r_p: 20}

    @property
    def default_geometry(self):
        return {"positive particle": {"r_p": {"min": 0, "max": 1}}}

    @property
    def default_submesh_types(self):
        return {"positive particle": pybamm.Uniform1DSubMesh}

    @property
    def default_spatial_methods(self):
        return {"positive particle": pybamm.FiniteVolume()}

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
        F = pybamm.constants.F.value
        T = param["Ambient temperature [K]"]
        alpha_p = param["Positive electrode active material volume fraction"]
        c_max_p = param["Maximum concentration in positive electrode [mol.m-3]"]
        L_p = param["Positive electrode thickness [m]"]
        R_p = param["Positive particle radius [m]"]
        D_p = param["Positive particle diffusivity [m2.s-1]"]

        # Get reference exchange current density [A.m-2]
        ce0 = param["Initial concentration in electrolyte [mol.m-3]"]
        j0_p = param.evaluate(
            param["Positive electrode exchange-current density [A.m-2]"](
                ce0, c_max_p / 2, c_max_p, T
            )
        )

        # Compute the cell area
        A = param["Electrode height [m]"] * param["Electrode width [m]"]

        # Compute the initial stoichiometry
        sto_p_init = (
            param["Initial concentration in positive electrode [mol.m-3]"] / c_max_p
        )

        # Grouped parameters
        Q_th_p = F * alpha_p * c_max_p * L_p * A / 3600
        tau_d_p = R_p**2 / D_p

        # Estimate the series resistance, neglecting conductivity losses
        RT_F = pybamm.constants.R.value * T / F
        Rct_typ = (RT_F * R_p) / (3 * alpha_p * L_p * A * j0_p)
        R0 = Rct_typ + param["Contact resistance [Ohm]"]

        parameter_dictionary = {
            "Nominal cell capacity [A.h]": param["Nominal cell capacity [A.h]"],
            "Current function [A]": param["Current function [A]"],
            "Initial stoichiometry": sto_p_init,
            "Lower voltage cut-off [V]": param["Lower voltage cut-off [V]"],
            "Upper voltage cut-off [V]": param["Upper voltage cut-off [V]"],
            "Positive electrode OCP [V]": param["Positive electrode OCP [V]"],
            "Theoretical electrode capacity [A.h]": Q_th_p,
            "Positive particle diffusion time scale [s]": tau_d_p,
            "Series resistance [Ohm]": R0,
        }
        parameter_values = ParameterValues(values=parameter_dictionary)
        parameter_values._set_initial_state = SPDiffusion.set_initial_state  # noqa: SLF001
        return parameter_values
