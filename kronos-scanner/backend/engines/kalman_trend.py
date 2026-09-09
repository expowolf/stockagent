# engines/kalman_trend.py
"""Kalman filter for dynamic trend extraction and alpha estimation."""

import numpy as np


class KalmanTrendFilter:
    """
    Local-linear-trend state-space model.

    State:        [price_level, trend_velocity]
    Observation:  raw price
    F (transition): [[1, 1], [0, 1]]  (constant-velocity model)
    H (observation): [1, 0]           (observe level only)
    """

    def __init__(self, process_variance=0.01, measurement_variance=1.0):
        self.Q = np.array([[process_variance, 0.0], [0.0, process_variance]])
        self.R = float(measurement_variance)
        self.H = np.array([[1.0, 0.0]])
        self.F = np.array([[1.0, 1.0], [0.0, 1.0]])

        self.x = np.array([0.0, 0.0])  # [level, velocity]
        self.P = np.eye(2) * 1.0
        self._initialized = False

    def update(self, z):
        """
        One Kalman predict/update step for observed price `z`.
        Returns: (trend_level, velocity, innovation_residual).
        """
        # Lazy init: seed the level with the first observation so the filter
        # does not spend many steps catching up from zero.
        if not self._initialized:
            self.x = np.array([float(z), 0.0])
            self.P = np.eye(2) * self.R
            self._initialized = True

        # Predict
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # Update
        y = float(z) - float((self.H @ x_pred)[0])          # innovation
        S = float((self.H @ P_pred @ self.H.T)[0, 0]) + self.R  # innovation cov
        K = (P_pred @ self.H.T).flatten() / (S + 1e-9)      # Kalman gain (2,)

        self.x = x_pred + K * y
        self.P = (np.eye(2) - np.outer(K, self.H)) @ P_pred

        return float(self.x[0]), float(self.x[1]), y

    def smooth_series(self, prices):
        """
        Forward-filter a price series.
        Returns: (trends, residuals, alpha) where alpha is the mean innovation
        normalized by the latest price (a dimensionless drift proxy).
        """
        prices = np.asarray(prices, dtype=float)
        prices = prices[~np.isnan(prices)]

        trends = []
        residuals = []

        for price in prices:
            trend, _vel, resid = self.update(price)
            trends.append(trend)
            residuals.append(resid)

        residuals = np.asarray(residuals)
        # Normalize so alpha is comparable across tickers of different price levels.
        denom = prices[-1] if len(prices) and prices[-1] != 0 else 1.0
        alpha = float(np.mean(residuals) / denom) if len(residuals) else 0.0

        return np.asarray(trends), residuals, alpha
