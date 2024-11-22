from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import scipy.linalg as linalg

from pybop._dataset import Dataset
from pybop.models.base_model import BaseModel, Inputs
from pybop.observers.observer import Observer
from pybop.parameters.parameter import Parameter


class UnscentedKalmanFilterObserver(Observer):
    """
    An observer using the unscented Kalman filter. This is a wrapper class for PyBOP, see class SquareRootUKF for more details on the method.

    Parameters
    ----------
    parameters: Parameters
        The parameters for the model.
    model : BaseModel
        The model to observe.
    sigma0 : np.ndarray | float
        The covariance matrix of the initial state. If a float is provided, the covariance matrix is set to sigma0 * np.eye(n), where n is the number of states.
        To remove a state from the filter, set the corresponding row and col to zero in both sigma0 and process.
    process : np.ndarray | float
        The covariance matrix of the process noise. If a float is provided, the covariance matrix is set to process * np.eye(n), where n is the number of states.
        To remove a state from the filter, set the corresponding row and col to zero in both sigma0 and process.
    measure : np.ndarray | float
        The covariance matrix of the measurement noise. If a float is provided, the covariance matrix is set to measure * np.eye(m), where m is the number of measurements.
    dataset : Dataset
        Dataset object containing the data to fit the model to.
    check_model : bool, optional
        Flag to indicate if the model should be checked (default: True).
    signal: str
        The signal to observe.
    initial_state : dict, optional
        A valid initial state, e.g. the initial open-circuit voltage (default: None).
    """

    Covariance = np.ndarray

    def __init__(
        self,
        parameters: list[Parameter],
        model: BaseModel,
        sigma0: Union[Covariance, float],
        process: Union[Covariance, float],
        measure: Union[Covariance, float],
        dataset: Optional[Dataset] = None,
        check_model: bool = True,
        signal: Optional[list[str]] = None,
        additional_variables: Optional[list[str]] = None,
        initial_state: Optional[float] = None,
    ) -> None:
        if model is not None:
            # Clear any existing built model and its properties
            if model.built_model is not None:
                model.clear()

            # Build the model from scratch
            model.build(
                dataset=dataset,
                parameters=parameters,
                check_model=check_model,
            )

        super().__init__(
            parameters, model, check_model, signal, additional_variables, initial_state
        )
        if dataset is not None:
            # Check that the dataset contains necessary variables
            dataset.check(signal=[*self.signal, "Current function [A]"])

            self._dataset = dataset.data
            self._domain_data = self._dataset["Time [s]"]
            self.n_data = len(self._domain_data)
            self._target = {signal: self._dataset[signal] for signal in self.signal}

        # Observer initiation
        self._process = process

        x0 = self.get_current_state().as_ndarray()
        m0 = self.get_current_measure()

        m = len(m0)
        n = len(x0)
        if isinstance(sigma0, float):
            sigma0 = sigma0 * np.eye(n)
        if isinstance(process, float):
            process = process * np.eye(n)
        if isinstance(measure, float):
            measure = measure * np.eye(m)

        if sigma0.shape != (n, n):
            raise ValueError(f"sigma0 must be a square matrix of size n = {n}")
        if process.shape != (n, n):
            raise ValueError(f"process must be a square matrix of size n = {n}")
        if measure.shape != (m, m):
            raise ValueError(f"measure must be a square matrix of size m = {m}")

        self._sigma0 = sigma0

        def measure_f(x: np.ndarray) -> np.ndarray:
            x = x.reshape(-1, 1)
            sol = self._model.get_state(inputs=self._state.inputs, t=self._state.t, x=x)
            return self.get_measure(sol).reshape(-1)

        self._ukf = SquareRootUKF(
            x0=x0,
            P0=sigma0,
            Rp=process,
            Rm=measure,
            f=None,
            h=measure_f,
        )

    def reset(self, inputs: Inputs) -> None:
        super().reset(inputs)
        self._ukf.reset(self.get_current_state().as_ndarray(), self._sigma0)

    def observe(self, time: float, value: np.ndarray) -> float:
        if value is None:
            raise ValueError("Measurement must be provided.")
        elif isinstance(value, np.floating):
            value = np.asarray([value])

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
        log_likelihood = self._ukf.step(value)
        self._state = self._model.get_state(
            inputs=self._state.inputs, t=time, x=self._ukf.x
        )
        return log_likelihood

    def get_current_covariance(self) -> Covariance:
        # Get the covariance from the square-root covariance
        return self._ukf.S @ self._ukf.S.T


@dataclass
class SigmaPoint:
    """
    A sigma point is a point in the state space that is used to estimate the mean and covariance of a random variable.
    """

    x: np.ndarray
    w_m: float
    w_c: float


class SquareRootUKF:
    """
    van der Menve, R., & Wan, E. A. (2001). THE SQUARE-ROOT UNSCENTED KALMAN FILTER FOR STATE AND PARAMETER-ESTIMATION.
    https://doi.org/10.1109/ICASSP.2001.940586

    We implement a square root unscented Kalman filter (UKF) with additive process and measurement noise.

    The square root UKF is a variant of the UKF that is more numerically stable and has better performance.

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
        # Find states that are zero in both sigma0 and process
        zero_rows = np.logical_and(np.all(P0 == 0, axis=0), np.all(Rp == 0, axis=0))
        zero_cols = np.logical_and(np.all(P0 == 0, axis=1), np.all(Rp == 0, axis=1))
        zeros = np.logical_and(zero_rows, zero_cols)
        ones = np.logical_not(zeros)
        states = np.asarray(range(len(x0)))[ones]
        bool_mask = np.ix_(ones, ones)

        S_filtered = linalg.cholesky(P0[ones, :][:, ones])
        sqrtRp_filtered = linalg.cholesky(Rp[ones, :][:, ones])

        n = len(x0)
        S = np.zeros((n, n))
        sqrtRp = np.zeros((n, n))
        S[bool_mask] = S_filtered
        sqrtRp[bool_mask] = sqrtRp_filtered

        self.x = x0
        self.S = S
        self.sqrtRp = sqrtRp
        self.sqrtRm = linalg.cholesky(Rm)
        self.alpha = 1e-3
        self.beta = 2
        self.f = f
        self.h = h
        self.states = states
        self.bool_mask = bool_mask

    def reset(self, x: np.ndarray, S: np.ndarray) -> None:
        self.x = x
        S_filtered = S[self.states, :][:, self.states]
        S_filtered = linalg.cholesky(S_filtered)
        S_full = S.copy()
        S_full[self.bool_mask] = S_filtered
        self.S = S_full

    @staticmethod
    def gen_sigma_points(
        x: np.ndarray, S: np.ndarray, alpha: float, beta: float, states: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates 2L+1 sigma points for the unscented transform, where L is the number of states.

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
        states: np.ndarray
            array of indices of states to use for the sigma points

        Returns
        -------
        list[np.ndarray]
            The sigma points
        list[float]
            The weights of the sigma points
        list[float]
            The weights of the covariance of the sigma points
        """
        # Set the scaling parameters: sigma and eta
        kappa = 0.0
        L = len(states)
        sigma = alpha**2 * (L + kappa) - L
        eta = np.sqrt(L + sigma)

        # Define the sigma points
        points = np.hstack(
            [x]
            + [x + eta * S[:, i].reshape(-1, 1) for i in states]
            + [x - eta * S[:, i].reshape(-1, 1) for i in states]
        )

        # Define the weights of the sigma points
        w_m0 = sigma / (L + sigma)
        w_m = np.asarray([w_m0] + [1 / (2 * (L + sigma))] * (2 * L))

        # Define the weights of the covariance of the sigma points
        w_c0 = w_m0 + (1 - alpha**2 + beta)
        w_c = np.asarray([w_c0] + [1 / (2 * (L + sigma))] * (2 * L))

        return (points, w_m, w_c)

    @staticmethod
    def unscented_transform(
        sigma_points: np.ndarray,
        w_m: np.ndarray,
        w_c: np.ndarray,
        sqrtR: np.ndarray,
        states: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Performs the unscented transform

        Parameters
        ----------
        sigma_points : list[SigmaPoint]
            The sigma points
        sqrtR : np.ndarray
            The square root of the covariance matrix
        states: np.ndarray
            array of indices of states to use for the transform

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The mean and square-root covariance of the sigma points
        """
        # Update the predicted mean of the sigma points
        x = np.sum(w_m * sigma_points, axis=1).reshape(-1, 1)

        # Update the predicted square-root covariance
        if states is None:
            sigma_points_diff = sigma_points - x
            A = np.hstack([np.sqrt(w_c[1:]) * (sigma_points_diff[:, 1:]), sqrtR])
            (_, S) = linalg.qr(A.T, mode="economic")
            S = SquareRootUKF.cholupdate(S, sigma_points_diff[:, 0:1], w_c[0])
        else:
            # First overwrite states without noise to remove numerial error
            clean = np.full(len(x), True)
            clean[states] = False
            x[clean] = sigma_points[clean, 0].reshape(-1, 1)

            sigma_points_diff = sigma_points[states, :] - x[states]
            A = np.hstack(
                [
                    np.sqrt(w_c[1:]) * (sigma_points_diff[:, 1:]),
                    sqrtR[states, :][:, states],
                ]
            )
            (_, S_filtered) = linalg.qr(A.T, mode="economic")
            S_filtered = SquareRootUKF.cholupdate(
                S_filtered, sigma_points_diff[:, 0:1], w_c[0]
            )
            ones = np.logical_not(clean)
            S = np.zeros_like(sqrtR)
            S[np.ix_(ones, ones)] = S_filtered

        return x, S

    @staticmethod
    def filtered_cholupdate(
        R: np.ndarray, x: np.ndarray, w: float, states: np.ndarray
    ) -> np.ndarray:
        R_full = R.copy()
        R_filtered = R[states, :][:, states]
        x_filtered = x[states]
        R_filtered = SquareRootUKF.cholupdate(R_filtered, x_filtered, w)
        ones = np.full(len(x), False)
        ones[states] = True
        R_full[np.ix_(ones, ones)] = R_filtered
        return R_full

    @staticmethod
    def cholupdate(R: np.ndarray, x: np.ndarray, w: float) -> np.ndarray:
        """
        Updates the Cholesky decomposition of a matrix (see https://github.com/modusdatascience/choldate/blob/master/choldate/_choldate.pyx)

        Note: will be in scipy soon so replace with this: https://github.com/scipy/scipy/pull/16499

        TODO: need to replace with something more low-level

        Parameters
        ----------
        R : np.ndarray
            The Cholesky decomposition of the matrix
        x : np.ndarray
            The vector to add to the matrix
        w : float
            The weight of the vector

        Returns
        -------
        np.ndarray
            The updated Cholesky decomposition
        """
        sign = np.sign(w)
        x = np.sqrt(abs(w)) * x.flatten()
        p = x.shape[0]
        for k in range(p):
            Rkk = abs(R[k, k])
            xk = abs(x[k])
            r = SquareRootUKF.hypot(Rkk, xk, sign)
            c = r / R[k, k]
            s = x[k] / R[k, k]
            R[k, k] = r
            if k < p - 1:
                R[k, k + 1 :] = (R[k, k + 1 :] + sign * s * x[k + 1 :]) / c
                x[k + 1 :] = c * x[k + 1 :] - s * R[k, k + 1 :]
        return R

    def hypot(R: float, x: float, sign: float) -> float:
        if R < x:
            return R * np.sqrt(1 + sign * R**2 / x**2)
        elif x < R:
            return np.sqrt(R**2 + sign * x**2)
        else:
            return 0.0

    def step(self, y: np.ndarray) -> float:
        """
        Steps the filter forward one step using a measurement. Returns the log likelihood of the measurement.

        Parameters
        ----------
        y : np.ndarray
            The measurement vector

        Returns
        -------
        float
            The log likelihood of the measurement
        """
        # Sigma point calculation
        sigma_points, w_m, w_c = self.gen_sigma_points(
            self.x, self.S, self.alpha, self.beta, self.states
        )

        # Update sigma points in time
        sigma_points = np.apply_along_axis(self.f, 0, sigma_points)

        # Compute the mean and square-root covariance
        x_minus, S_minus = self.unscented_transform(
            sigma_points, w_m, w_c, self.sqrtRp, self.states
        )

        # Compute the output corresponding to the updated sigma points
        sigma_points_y = np.apply_along_axis(self.h, 0, sigma_points)

        # Compute the mean and square-root covariance
        y_minus, S_y = self.unscented_transform(sigma_points_y, w_m, w_c, self.sqrtRm)

        # Compute the gain from the covariance
        P = np.einsum(
            "k,jk,lk -> jl ", w_c, sigma_points - x_minus, sigma_points_y - y_minus
        )
        gain = linalg.lstsq(linalg.lstsq(P.T, S_y.transpose())[0].T, S_y)[0]

        # Update the states and square-root covariance based on the gain
        residual = y - y_minus
        self.x = x_minus + gain @ residual
        U = gain @ S_y
        self.S = self.filtered_cholupdate(S_minus, U, -1, self.states)

        # Compute the log-likelihood of the covariance
        S = self.S[self.states, :][:, self.states]
        log_det = 2 * np.sum(np.log(np.diag(S)))
        n = len(y)
        log_likelihood = -0.5 * (
            n * log_det + residual.T @ linalg.cho_solve((S_y, True), residual)
        )
        return np.sum(log_likelihood)
