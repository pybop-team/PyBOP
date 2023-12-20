from dataclasses import dataclass
import numpy as np
import scipy.linalg as linalg
from typing import Tuple

from pybop.models.base_model import BaseModel, Inputs
from pybop.observers.observer import Observer


class UnscentedKalmanFilterObserver(Observer):
    """
    An observer using the unscented kalman filter. This is a wrapper class for PyBOP, see class UkfFilter for more details on the method.

    Parameters
    ----------
    model : BaseModel
        The model to observe.
    inputs: Dict[str, float]
        The inputs to the model.
    signal: str
        The signal to observe.
    sigma0 : np.ndarray
        The covariance matrix of the initial state.
    process : np.ndarray
        The covariance matrix of the process noise.
    measure : np.ndarray
        The covariance matrix of the measurement noise.
    """

    Covariance = np.ndarray

    def __init__(
        self,
        model: BaseModel,
        inputs: Inputs,
        signal: str,
        sigma0: Covariance,
        process: Covariance,
        measure: Covariance,
    ) -> None:
        super().__init__(model, inputs, signal)
        self._process = process

        x0 = self.get_current_state().as_ndarray()
        m0 = self.get_current_measure()

        m = len(m0)
        n = len(x0)
        if sigma0.shape != (n, n):
            raise ValueError(f"sigma0 must be a square matrix of size n = {n}")
        if process.shape != (n, n):
            raise ValueError(f"process must be a square matrix of size n = {n}")
        if measure.shape != (m, m):
            raise ValueError(f"measure must be a square matrix of size m = {m}")

        def measure_f(x: np.ndarray) -> np.ndarray:
            x = x.reshape(-1, 1)
            sol = self._model.get_state(inputs=self._state.inputs, t=self._state.t, x=x)
            return self.get_measure(sol).reshape(-1)

        self._ukf = UkfFilter(
            x0=x0,
            P0=sigma0,
            Rp=process,
            Rm=measure,
            f=None,
            h=measure_f,
        )

    def observe(self, time: float, value: np.ndarray | None = None) -> None:
        if value is None:
            raise ValueError("Measurement must be provided.")

        dt = time - self.get_current_time()
        if dt < 0:
            raise ValueError("Time must be increasing.")

        if dt == 0:

            def f(x: np.ndarray) -> np.ndarray:
                return x
        else:

            def f(x: np.ndarray) -> np.ndarray:
                x = x.reshape(-1, 1)
                sol = self._model.get_state(
                    inputs=self._state.inputs, t=self._state.t, x=x
                )
                return self._model.step(sol, time).as_ndarray().reshape(-1)

        self._ukf.f = f
        self._ukf.Rp = dt * self._process
        self._ukf.step(value)
        self._state = self._model.get_state(
            inputs=self._state.inputs, t=time, x=self._ukf.x
        )


@dataclass
class SigmaPoint(object):
    """
    A sigma point is a point in the state space that is used to estimate the mean and covariance of a random variable.
    """

    x: np.ndarray
    w_m: float
    w_c: float


class UkfFilter(object):
    """
    van der Menve, R., & Wan, E. A. (n.d.). THE SQUARE-ROOT UNSCENTED KALMAN FILTER FOR STATE AND PARAMETER-ESTIMATION. http://wol.ra.phy.cam.ac.uk/mackay

    we implement a UKF filter with additive process and measurement noise

    the square root unscented kalman filter is a variant of the unscented kalman filter that is more numerically stable and has better performance.

    Parameters
    ----------
    x0 : np.ndarray
        The initial state vector
    P0 : np.ndarray
        The initial covariance matrix
    Rp : np.ndarray
        The covariance matrix of the process noise
    Rm : np.ndarray
        The covariance matrix of the measurement noise
    f : callable
        The state transition function
    h : callable
        The measurement function
    """

    def __init__(
        self,
        x0: np.ndarray,
        P0: np.ndarray,
        Rp: np.ndarray,
        Rm: np.ndarray,
        f: callable,
        h: callable,
    ) -> None:
        self.x = x0
        self.S = linalg.cholesky(P0)
        self.sqrtRp = linalg.cholesky(Rp)
        self.sqrtRm = linalg.cholesky(Rm)
        self.alpha = 1e-3
        self.beta = 2
        self.f = f
        self.h = h

    @staticmethod
    def gen_sigma_points(
        x: np.ndarray, S: np.ndarray, alpha: float, beta: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates the sigma points for the unscented transform

        Parameters
        ----------
        x : np.ndarray
            The state vector
        S : np.ndarray
            The square root of the covariance matrix
        alpha : float
            The spread of the sigma points. Typically 1e-4 < alpha < 1
        beta : float
            The prior knowledge of the distribution. Typically 2 for a Gaussian distribution

        Returns
        -------
        List[np.ndarray]
            The sigma points
        List[float]
            The weights of the sigma points
        List[float]
            The weights of the covariance points
        """
        kappa = 1.0
        L = len(x)
        sigma = alpha**2 * (L + kappa) - L
        eta = np.sqrt(L + sigma)
        wm_0 = sigma / (L + sigma)
        wc_0 = wm_0 + (1 - alpha**2 + beta)
        points = np.hstack(
            [x]
            + [x + eta * S[:, i].reshape(-1, 1) for i in range(L)]
            + [x - eta * S[:, i].reshape(-1, 1) for i in range(L)]
        )
        w_m = np.array([wm_0] + [(1 - wm_0) / (2 * L)] * (2 * L))
        w_c = np.array([wc_0] + [(1 - wc_0) / (2 * L)] * (2 * L))
        return (points, w_m, w_c)

    @staticmethod
    def unscented_transform(
        sigma_points: np.ndarray, w_m: np.ndarray, w_c: np.ndarray, sqrtR: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs the unscented transform

        Parameters
        ----------
        sigma_points : List[SigmaPoint]
            The sigma points
        sqrtR : np.ndarray
            The square root of the covariance matrix

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The mean and covariance of the sigma points
        """
        x = np.sum(w_m * sigma_points, axis=1).reshape(-1, 1)
        A = np.sqrt(w_c[1:]) * (sigma_points[:, 1:] - x)
        (
            _,
            S,
        ) = linalg.qr(A.T, mode="economic")
        S = UkfFilter.cholupdate(S, sigma_points[:, 0:1] - x, w_c[0])
        return x, S

    @staticmethod
    def hypot(x: float, y: float) -> float:
        x = abs(x)
        y = abs(y)
        t = x if x < y else y
        x = x if x > y else y
        t = t / x
        return x * np.sqrt(1 + t * t)

    @staticmethod
    def cholupdate(R: np.ndarray, x: np.ndarray, w: float) -> np.ndarray:
        """
        Updates the cholesky decomposition of a matrix (see https://github.com/modusdatascience/choldate/blob/master/choldate/_choldate.pyx)

        Note: will be in scipy soon so replace with this: https://github.com/scipy/scipy/pull/16499

        TODO: need to replace with something more low-level

        Parameters
        ----------
        R : np.ndarray
            The cholesky decomposition of the matrix
        x : np.ndarray
            The vector to add to the matrix
        w : float
            The weight of the vector

        Returns
        -------
        np.ndarray
            The updated cholesky decomposition
        """
        x = x.flatten()
        p = x.shape[0]
        for k in range(p):
            r = np.sqrt(R[k, k] ** 2 + w * x[k] ** 2)
            # r = UkfFilter.hypot(R[k, k], x[k])
            c = r / R[k, k]
            s = x[k] / R[k, k]
            R[k, k] = r
            if k < p - 1:
                R[k, k + 1 :] = (R[k, k + 1 :] + w * s * x[k + 1 :]) / c
                x[k + 1 :] = c * x[k + 1 :] - s * R[k, k + 1 :]
        return R

    def step(self, y: np.ndarray) -> None:
        """
        Steps the filter forward one step using a measurement

        Parameters
        ----------
        y : np.ndarray
            The measurement vector
        """
        sigma_points, w_m, w_c = self.gen_sigma_points(
            self.x, self.S, self.alpha, self.beta
        )
        sigma_points = np.apply_along_axis(self.f, 0, sigma_points)

        x_minus, S_minus = self.unscented_transform(sigma_points, w_m, w_c, self.sqrtRp)
        sigma_points_y = np.apply_along_axis(self.h, 0, sigma_points)
        y_minus, S_y = self.unscented_transform(sigma_points_y, w_m, w_c, self.sqrtRm)
        P = np.sum(
            w_c * np.multiply.outer(sigma_points - x_minus, sigma_points_y - y_minus),
            axis=(1, 3),
        )
        gain = linalg.lstsq(linalg.lstsq(P.T, S_y.transpose())[0].T, S_y)[0]
        self.x = x_minus + gain @ (y - y_minus)
        U = gain @ S_y
        self.S = self.cholupdate(S_minus, U, -1)
