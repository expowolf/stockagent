# engines/ou_reversion.py
"""Ornstein-Uhlenbeck mean-reversion engine."""

import numpy as np


class OUMeanReversion:
    """
    Fit OU process: dx = theta(mu - x)dt + sigma dW
    Estimate theta, mu, sigma via OLS discretization on the price level.
    """

    def __init__(self, window=252):
        self.window = window
        self.theta = None
        self.mu = None
        self.sigma = None
        self.halflife = None

    def fit(self, prices):
        """
        Fit OU to the last `window` prices.
        Returns: (theta, mu, sigma, halflife) or None on failure.
        """
        prices = np.asarray(prices, dtype=float)
        prices = prices[~np.isnan(prices)]

        if len(prices) > self.window:
            prices = prices[-self.window:]

        if len(prices) < 3:
            return None

        mu_hat = float(np.mean(prices))

        # Discretized OU:  x_{t+1} - x_t = theta * (mu - x_t) * dt + eps
        # Regress dx on (mu - x_t).  dt is folded into the coefficient.
        X = np.vstack([np.ones(len(prices) - 1), prices[:-1] - mu_hat]).T
        y = prices[1:] - prices[:-1]

        try:
            coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            return None

        slope = coeffs[1]
        # Regressor is (x_t - mu); dx = theta(mu - x_t)dt = -theta(x_t - mu)dt,
        # so slope = -theta*dt and theta = -slope (dt folded in).
        theta = -slope

        if theta <= 1e-3:
            theta = 1e-3

        residuals = y - X @ coeffs
        sigma = float(np.std(residuals))

        self.theta = float(theta)
        self.mu = mu_hat
        self.sigma = sigma
        self.halflife = float(np.log(2) / theta)

        return self.theta, self.mu, self.sigma, self.halflife

    def score(self, current_price):
        """
        Mean-reversion score in [0, 100].
        Blends distance-from-mean (z-score) with half-life attractiveness.
        """
        if self.theta is None or self.mu is None or self.sigma is None:
            return 50.0

        distance_from_mean = abs(current_price - self.mu)
        z_score = distance_from_mean / (self.sigma + 1e-9)

        # Half-life sweet spot ~20 days (Gaussian bump).
        halflife_score = 100.0 * np.exp(-((self.halflife - 20.0) ** 2) / 200.0)

        # Distance score: clip extreme z-scores at 3 sigma.
        z_score_clipped = min(z_score, 3.0) / 3.0
        distance_score = 100.0 * z_score_clipped

        return float(0.6 * distance_score + 0.4 * halflife_score)
