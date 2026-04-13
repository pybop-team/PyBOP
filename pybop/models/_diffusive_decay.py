from numpy import asarray, cos, exp, pi, sin, zeros_like
from scipy.integrate import quad


class Diffusive_Relaxation:
    """Solution to ∂ₜ u = D ∂ₓ² u with u(x, t=0) = f(x)."""

    def __init__(self, f, L, summands=10, radial=False):
        self.f = f
        self.L = L
        self.summands = summands
        self.radial = radial
        if radial:
            pass
        else:
            self.series = cos
            self.coefficients = self.compute_zero_flow_coefficients()

    def compute_zero_flow_coefficients(self):
        coefficients = asarray(
            [
                2.0
                / self.L
                * quad(lambda x, n=n: self.f(x) * cos(n * pi * x / self.L), 0, self.L)[
                    0
                ]
                for n in range(0, self.summands)
            ]
        )
        # In order to use a simple summation expression suitable for
        # automatic differentiation, half the "zeroth" coefficient.
        coefficients[0] = coefficients[0] / 2.0
        return coefficients

    def compute_zero_concentration_coefficients(self):
        # In order to use the same summation expression for both cases,
        # the "zeroth" coefficient is set to 0.
        return asarray(
            [0]
            + [
                2.0
                / self.L
                * quad(lambda x, n=n: self.f(x) * sin(n * pi * x / self.L), 0, self.L)[
                    0
                ]
                for n in range(1, self.summands)
            ]
        )

    def concentration(self, x, t, D=1.0):
        value = zeros_like(D * t)
        for n in range(self.summands):
            value += (
                self.coefficients[n]
                * self.series(n * pi * x / self.L)
                * exp(-(n**2) * pi**2 * D * t / self.L**2)
            )
        return value

    def __call__(self, t, offset=0.0, timescale=1.0, magnitude=1.0):
        D = self.L**2 / timescale
        return offset + magnitude * (
            self.concentration(self.L, t, D)
            - self.concentration(self.L, t, D)
            - self.concentration(1.0, 0.0, D)
            + self.concentration(0.0, 0.0, D)
        )
