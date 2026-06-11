# engines/hmm_regime.py
"""Hidden Markov Model regime detector (Bull / Bear / HighVol / LowVol)."""

import numpy as np
import pandas as pd

try:
    # Canonical hmmlearn import path.
    from hmmlearn.hmm import GaussianHMM
except ImportError:  # pragma: no cover - fallback for older layouts
    from hmmlearn.gaussian_hmm import GaussianHMM


class HMMRegimeDetector:
    """
    Classify the market into 4 regimes via Baum-Welch EM on
    [log-return, rolling 20-day volatility] features.

    Regime labels are assigned by ranking the fitted state means rather
    than relying on raw state indices, so labels are semantically stable.
    """

    def __init__(self, window=252, n_states=4):
        self.window = window
        self.n_states = n_states
        self.model = None
        # Mapping from raw HMM state index -> human label.
        self.state_to_label = {}

    @staticmethod
    def _features(prices):
        returns = np.diff(np.log(prices))
        rolling_vol = (
            pd.Series(returns).rolling(20).std().bfill().fillna(0.0).values
        )
        return np.column_stack([returns, rolling_vol])

    def _assign_labels(self):
        """Map raw states to Bull/Bear/HighVol/LowVol from fitted means."""
        means = self.model.means_  # shape (n_states, 2): [mean_return, mean_vol]
        ret_mean = means[:, 0]
        vol_mean = means[:, 1]

        labels = {}
        # Highest-return state -> Bull, lowest-return -> Bear.
        bull = int(np.argmax(ret_mean))
        bear = int(np.argmin(ret_mean))
        labels[bull] = "Bull"
        labels[bear] = "Bear"

        # Of the remaining states, highest vol -> HighVol, other -> LowVol.
        remaining = [s for s in range(self.n_states) if s not in (bull, bear)]
        if remaining:
            hi = max(remaining, key=lambda s: vol_mean[s])
            labels[hi] = "HighVol"
            for s in remaining:
                if s != hi:
                    labels[s] = "LowVol"

        # Any state not covered (e.g. n_states != 4) gets a generic label.
        for s in range(self.n_states):
            labels.setdefault(s, f"State{s}")

        self.state_to_label = labels

    def fit(self, prices):
        """Fit the HMM. Returns the fitted model or None on failure."""
        prices = np.asarray(prices, dtype=float)
        prices = prices[~np.isnan(prices)]

        if len(prices) > self.window:
            prices = prices[-self.window:]

        if len(prices) < 50:
            return None

        features = self._features(prices)

        # Try progressively more robust covariance structures. "full" can fail
        # with non-positive-definite covariances on near-degenerate data, so we
        # fall back to "diag" / "spherical" before giving up.
        for cov_type in ("full", "diag", "spherical"):
            try:
                model = GaussianHMM(
                    n_components=self.n_states,
                    covariance_type=cov_type,
                    n_iter=1000,
                    random_state=42,
                    min_covar=1e-3,
                )
                model.fit(features)
                self.model = model
                self._assign_labels()
                return self.model
            except Exception as exc:  # noqa: BLE001 - try next cov type
                last_exc = exc
                continue

        print(f"HMM fitting failed: {last_exc}")
        self.model = None
        return None

    def predict_regime(self, prices):
        """
        Return (regime_name, regime_probability_vector, confidence).
        Confidence is the posterior probability of the most recent state.
        """
        if self.model is None:
            return "Unknown", np.ones(self.n_states) / self.n_states, 0.0

        prices = np.asarray(prices, dtype=float)
        prices = prices[~np.isnan(prices)]

        if len(prices) < 22:
            return "Unknown", np.ones(self.n_states) / self.n_states, 0.0

        features = self._features(prices)

        try:
            _, posteriors = self.model.score_samples(features)
            regime_probs = posteriors[-1]
            current_state = int(np.argmax(regime_probs))
            regime_name = self.state_to_label.get(current_state, f"State{current_state}")
            confidence = float(regime_probs[current_state])
            return regime_name, regime_probs, confidence
        except Exception:  # noqa: BLE001
            return "Unknown", np.ones(self.n_states) / self.n_states, 0.0
