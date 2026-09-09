# engines/evt_risk.py
"""Extreme Value Theory (Peak-Over-Threshold) tail-risk engine."""

import numpy as np
from scipy.stats import genpareto


class EVTTailRisk:
    """
    Peak-Over-Threshold EVT. Fit the negative-return tail to a Generalized
    Pareto Distribution (GPD) and derive VaR / Expected Shortfall.

    Convention: VaR and ES are returned as positive loss magnitudes
    (e.g. 0.03 == a 3% loss).
    """

    def __init__(self, threshold_percentile=90, window=252):
        # Percentile of |negative returns| used as the POT threshold.
        self.threshold_percentile = threshold_percentile
        self.window = window
        self.threshold = None
        self.shape = None  # xi
        self.scale = None  # sigma
        self.exceed_rate = None  # fraction of losses above threshold

    def fit(self, prices):
        """Fit GPD to the loss tail. Returns (shape, scale) or None."""
        prices = np.asarray(prices, dtype=float)
        prices = prices[~np.isnan(prices)]

        if len(prices) > self.window:
            prices = prices[-self.window:]

        if len(prices) < 20:
            return None

        returns = np.diff(np.log(prices))
        losses = -returns[returns < 0]  # positive loss magnitudes

        if len(losses) < 10:
            return None

        # POT threshold: high percentile of the loss distribution.
        self.threshold = float(np.percentile(losses, self.threshold_percentile))

        exceedances = losses[losses > self.threshold] - self.threshold
        if len(exceedances) < 5:
            return None

        # Fraction of ALL losses that exceed the threshold (for tail scaling).
        self.exceed_rate = float(np.mean(losses > self.threshold))

        try:
            shape, _loc, scale = genpareto.fit(exceedances, floc=0)
        except Exception:  # noqa: BLE001
            return None

        self.shape = float(shape)
        self.scale = float(scale)
        return self.shape, self.scale

    def calculate_var_es(self, tail_prob=0.01):
        """
        Compute VaR and Expected Shortfall at `tail_prob` (e.g. 0.01 == 99% VaR).
        Returns (var, es, risk_score). risk_score in [0, 100], higher = worse.
        """
        if self.shape is None or self.scale is None or not self.exceed_rate:
            return None, None, 50.0

        # GPD POT tail quantile (McNeil & Frey):
        #   VaR_p = u + (sigma/xi) * (((1 - p_target)/exceed_rate_tail)^-xi - 1)
        # Here we work directly with the exceedance rate over threshold u.
        ratio = tail_prob / self.exceed_rate
        ratio = max(ratio, 1e-12)

        xi = self.shape
        u = self.threshold
        sigma = self.scale

        if abs(xi) < 1e-6:
            var = u - sigma * np.log(ratio)
        else:
            var = u + (sigma / xi) * (ratio ** (-xi) - 1.0)

        # Expected Shortfall for the GPD tail (defined for xi < 1).
        if xi < 1.0:
            es = (var + (sigma - xi * u)) / (1.0 - xi)
        else:
            es = var  # heavy tail: ES undefined; fall back to VaR.

        # Risk score grows with tail-heaviness (xi) and the size of VaR.
        risk_score = float(min(100.0, 50.0 + 50.0 * max(xi, 0.0)))

        return float(var), float(es), risk_score
