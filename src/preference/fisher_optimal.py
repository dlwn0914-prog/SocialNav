"""Fisher-information-weighted D-optimal query selection -- the fix
motivated by utils/dpb_estimation_regret.py + the follow-up diagnosis:
ActiveQueryStrategy, DPBStrategy, and even plain DOptimalStrategy all
underperform random sampling at full preference-VECTOR estimation because
none of them account for the Bradley-Terry/logistic observation model's
Fisher information curvature. For a logistic model, the Fisher information
of a comparison with logit z = X^T theta* is mu(z)(1-mu(z)) (mu = sigmoid),
which is maximized at z=0 (a genuine toss-up) and collapses to ~0 as the
comparison becomes "obvious" (|z| large) -- so a selection rule that
chases "big, confident-looking gaps" (as both ActiveQuery's challenger
search and DPB's UCB bonus do) is choosing exactly the LEAST informative
comparisons for pinning down theta precisely, even though they look
decisive. Plain D-optimal design (doptimal.py, maximizing X^T V^-1 X with
no Fisher weighting) inherits the same blind spot because it is really a
LINEAR-model design criterion, not a logistic one.

This combines both terms using the CURRENT estimate theta_hat (so it needs
exploitation of theta_hat, unlike plain D-optimal, but in the OPPOSITE
direction of ActiveQuery/DPB: it seeks pairs theta_hat says are CLOSE to
a toss-up, not ones it says are lopsided):

    score(X) = mu(X^T theta_hat) * (1 - mu(X^T theta_hat)) * (X^T V^-1 X)

On the first query theta_hat is initialized to the zero vector, so
mu(0)=0.5 is the same constant for every candidate pair -- the criterion
reduces exactly to plain D-optimal (pure coverage) until there is any
preference estimate to speak of.
"""
import numpy as np

from src.preference.active_query import fit_logit_mle


def _mu(x):
    return 1.0 / (1.0 + np.exp(-x))


class FisherOptimalStrategy:
    def __init__(self, lam=1.0, norm_bound=5.0, rng=None):
        self.lam = lam
        self.norm_bound = norm_bound
        self.rng = rng or np.random.default_rng()
        self.diff_feats, self.labels = [], []
        self.theta_hat = None

    def _V(self, dim):
        V = self.lam * np.eye(dim)
        for X in self.diff_feats:
            V += np.outer(X, X)
        return V

    def select(self, feats):
        feats = np.asarray(feats, dtype=np.float64)
        K, dim = feats.shape
        theta = self.theta_hat if self.theta_hat is not None else np.zeros(dim)
        V = self._V(dim)
        V_inv = np.linalg.inv(V)

        best_score, best_pair = -np.inf, (0, 1)
        for i in range(K):
            for j in range(i + 1, K):
                X = feats[i] - feats[j]
                p = _mu(float(X @ theta))
                fisher = p * (1 - p)
                coverage = float(X @ V_inv @ X)
                score = fisher * coverage
                if score > best_score:
                    best_score, best_pair = score, (i, j)
        return best_pair

    def update(self, feats, i, j, winner):
        diff = feats[i] - feats[j] if winner == i else feats[j] - feats[i]
        self.diff_feats.append(diff)
        self.labels.append(1.0)
        try:
            self.theta_hat = fit_logit_mle(np.array(self.diff_feats), np.array(self.labels), norm_bound=self.norm_bound)
        except RuntimeError:
            pass

    def best(self, feats):
        if self.theta_hat is None:
            return int(self.rng.integers(len(feats)))
        return int(np.argmax(np.asarray(feats) @ self.theta_hat))

    def done(self):
        return False
