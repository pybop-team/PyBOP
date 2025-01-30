from scipy import interpolate


class BaseApplication:
    """
    A base class for PyBOP's application methods.
    """

    def interp1d(self, x, y):
        return interpolate.interp1d(
            x,
            y,
            bounds_error=False,
            fill_value="extrapolate",
            axis=0,
        )
