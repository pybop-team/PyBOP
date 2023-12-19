from dataclasses import dataclass
import numpy as np
import scipy.linalg as linalg
from typing import Callable, Dict, List, Tuple

from pybop.models.base_model import BaseModel, Inputs, TimeSeriesState
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

        self._ukf = UkfFilter(
            x0=self._model.reinit(inputs).as_ndarray(),
            P0=sigma0,
            Rp=process,
            Rm=measure,
            f=self._model.step,
            h=self.get_current_measure,
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
                sol = self._model.reinit(
                    inputs=self._state.inputs, t=self._state.t, x=x
                )
                return self._model.step(sol, time).as_ndarray()

        self._ukf.f = f
        self._ukf.Rp = dt * self._process
        self._ukf.step(value)


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
        self.sqrtR = linalg.cholesky(Rp)
        self.sqrtRm = linalg.cholesky(Rm)
        self.alpha = 1e-3
        self.beta = 2
        self.f = f
        self.h = h

    @staticmethod
    def gen_sigma_points(
        x: np.ndarray, S: np.ndarray, alpha: float, beta: float
    ) -> List[SigmaPoint]:
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
        L = len(x)
        sigma = L * alpha**2 - 1
        eta = np.sqrt(L + sigma)
        return (
            [
                SigmaPoint(
                    x=x,
                    w_m=sigma / (L + sigma),
                    w_c=sigma / (L + sigma) + (1 - alpha**2 + beta),
                )
            ]
            + [
                SigmaPoint(
                    x=x + eta * S[:, i],
                    w_m=1 / (2 * (L + sigma)),
                    w_c=1 / (2 * (L + sigma)),
                )
                for i in range(S.shape[1])
            ]
            + [
                SigmaPoint(
                    x=x - eta * S[:, i],
                    w_m=1 / (2 * (L + sigma)),
                    w_c=1 / (2 * (L + sigma)),
                )
                for i in range(S.shape[1])
            ]
        )

    @staticmethod
    def unscented_transform(
        sigma_points: List[SigmaPoint], sqrtR: np.ndarray
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
        x = np.sum(
            [sigma_point.w_m * sigma_point.x for sigma_point in sigma_points], axis=0
        )
        s1_2L_concat = np.hstack([sigma_point.x for sigma_point in sigma_points])
        S = linalg.qr(
            np.hstack([sigma_points[1].w_c * (s1_2L_concat - x), sqrtR]), mode="r"
        )
        S = UkfFilter.cholupdate(S, sigma_points[0].x - x, sigma_points[0].w_c)
        return x, S

    @staticmethod
    def cholupdate(R: np.ndarray, x: np.ndarray, w: float) -> np.ndarray:
        """
        Updates the cholesky decomposition of a matrix

        TODO: no idea if this works, copilot wrote it, need to replace with something more low-level

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
        p = x.shape[0]
        for k in range(p):
            r = np.sqrt(R[k, k] ** 2 + w * x[k] ** 2)
            c = r / R[k, k]
            s = x[k] / R[k, k]
            R[k, k] = r
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
        sigma_points = self.gen_sigma_points(self.x, self.S, self.alpha, self.beta)
        sigma_points = [
            SigmaPoint(
                x=self.f(sigma_point.x), w_m=sigma_point.w_m, w_c=sigma_point.w_c
            )
            for sigma_point in sigma_points
        ]
        x_minus, S_minus = self.unscented_transform(sigma_points, self.sqrtRp)

        sigma_points_y = [
            SigmaPoint(
                x=self.h(sigma_point.x), w_m=sigma_point.w_m, w_c=sigma_point.w_c
            )
            for sigma_point in sigma_points
        ]
        y_minus, S_y = self.unscented_transform(sigma_points_y, self.sqrtRm)

        P = np.sum(
            [
                sigma_point.w_c
                * np.outer(sigma_point.x - x_minus, sigma_point_y.x - y_minus)
                for sigma_point, sigma_point_y in zip(sigma_points, sigma_points_y)
            ],
            axis=0,
        )
        gain = linalg.lstsq(linalg.lstsq(P, S_y.transpose()), S_y)
        self.x = x_minus + gain @ (y - y_minus)
        U = gain @ S_y
        self.S = self.cholupdate(S_minus, U, -1)
