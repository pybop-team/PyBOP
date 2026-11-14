from pybamm import (
    DummySolver,
    Parameter,
    ParameterValues,
    citations,
    constants,
    exp,
    log,
    tanh,
)
from pybamm import t as pybamm_t
from pybamm.models.base_model import BaseModel

from pybop.models._diffusive_decay import Diffusive_Relaxation


class SiliconRelaxation(BaseModel):
    """
    Calculates the mechanical voltage response from silicon at rest.

    This model from Koebbing et al. is a core-shell (silicon-SEI/oxide) model.
    """

    def __init__(self, name="Silicon voltage model after Koebbing2024", **kwargs):
        super().__init__(name=name, **kwargs)
        self._summary_variables = []

        citations.register("""@article{
            Köbbing2024,
            title={{Slow Voltage Relaxation of Silicon Nanoparticles with a Chemo-Mechanical Core-Shell Model}},
            author={Köbbing, L and Kuhn, Y and Horstmann, B},
            journal={ACS Applied Materials & Interfaces},
            volume={16},
            pages={67609-67619},
            year={2024}
        }""")

        ##############
        # Parameters #
        ##############

        # Parameters that match onto existing PyBaMM notation.
        R_core = Parameter("Negative particle radius [m]")
        # D = Parameter("Negative particle diffusivity [m]")
        L_shell = Parameter("Initial SEI thickness [m]")
        # T = Parameter("Ambient temperature [K]")
        F = constants.F
        # R = constants.R
        # New parameters for the mechanical properties.
        E_core = Parameter("Negative particle Young modulus [Pa]")
        # nu_core = Parameter("Negative particle Poisson ratio")
        # sigma_Y_core = Parameter("Negative particle yield stress [Pa]")
        v_Li = Parameter("Negative particle partial molar volume [m3.mol-1]")
        # E_shell = Parameter("Negative shell Young modulus [Pa]")
        # nu_shell = Parameter("Negative shell Poisson ratio")
        # sigma_Y_shell = Parameter("Negative shell yield stress [Pa]")
        # eta_shell = Parameter("Negative shell Newtonian viscosity [Pa.s]")

        # G_core = E_core / (2 * (1 + nu_core))  # Second Lame constant.
        # lambda_core = 2 * G_core * nu_core / (2 * (1 + nu_core))  # First Lame constant.

        alpha = 0.5 * (R_core / L_shell - 1)

        U_infty = Parameter("Terminal voltage of observed relaxation [V]")
        slope = Parameter("Logarithmic slope of observed mechanical relaxation [V]")
        timescale_exp = Parameter(
            "Exponential timescale for decay of observed mechanical relaxation [s]"
        )

        # Express the effective parameters from above by adjusting the parameters the model is written in.
        lambda_ch = (
            1  # lumped parameter of the model, set to 1 without loss of generality
        )
        sigma_ref = slope * alpha * F * lambda_ch**3 / (2 * v_Li)
        tau = lambda_ch * E_core * alpha * timescale_exp / sigma_ref

        # Optionally add a diffusive relaxation.
        if kwargs.get("with diffusion"):
            diff_portion = Parameter(
                "Fraction of observed relaxation that is diffusive"
            )
            diff_timescale = Parameter("Timescale of observed diffusive relaxation [s]")
            rel_magnitude = (1 - diff_portion) * U_infty
            diff_magnitude = diff_portion * U_infty
        else:
            rel_magnitude = U_infty
            diff_magnitude = 0

        # Determine integration constant from t=0 with
        # sigma_ev(t=0) = sigma_0 from sigma_ev = delta_U * F / v_Li.
        integration_constant = tanh(
            rel_magnitude * alpha * F * lambda_ch**3 / (2 * v_Li * sigma_ref)
        )

        # As PyBaMM has no arctanh, write it via arctanh = 0.5 * log((1 + x) / (1 - x)).
        arctanh_argument = integration_constant * exp(
            -E_core * alpha * lambda_ch / (tau * sigma_ref) * pybamm_t
        )
        delta_U = (
            2
            * v_Li
            * sigma_ref
            / (alpha * F * lambda_ch**3)
            * 0.5
            * log((1 + arctanh_argument) / (1 - arctanh_argument))
        )

        if kwargs.get("with diffusion"):
            delta_U += Diffusive_Relaxation(lambda x: x, L=1)(
                pybamm_t, -diff_magnitude, diff_timescale, diff_magnitude
            )

        self.variables = {
            "Voltage [V]": delta_U - U_infty,
            "Time [s]": pybamm_t,
            "Current [A]": 0,
            "Current variable [A]": 0,
        }

    @property
    def default_geometry(self):
        return {}

    @property
    def default_parameter_values(self) -> ParameterValues:
        return ParameterValues(
            {
                "Negative particle radius [m]": 30e-9,
                "Negative particle diffusivity [m]": 1e-17,
                "Initial SEI thickness [m]": 20e-9,
                "Ambient temperature [K]": 298,
                "Negative particle Young modulus [Pa]": 200e9,
                "Negative particle Poisson ratio": 0.22,
                "Negative particle yield stress [Pa]": 3e9,
                "Negative particle partial molar volume [m3.mol-1]": 9e-6,
                "Negative shell Young modulus [Pa]": 100e9,
                "Negative shell Poisson ratio": 0.3,
                "Negative shell yield stress [Pa]": 2e9,
                "Negative shell Newtonian viscosity [Pa.s]": 135e12,
                "Terminal voltage of observed relaxation [V]": 0.12,
                "Logarithmic slope of observed mechanical relaxation [V]": 0.025,
                "Exponential timescale for decay of observed mechanical relaxation [s]": 2e4,
                "Fraction of observed relaxation that is diffusive": 0.0,
                "Timescale of observed diffusive relaxation [s]": 90.0,
            }
        )

    @property
    def default_quick_plot_variables(self):
        return [
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
