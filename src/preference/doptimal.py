"""D-optimal query selection: a coverage-first alternative to
ActiveQueryStrategy (UserAlign-style) and DPBStrategy (Kim et al. 2024),
motivated directly by utils/dpb_estimation_regret.py's finding that both
of those collapse to an ill-conditioned design matrix (ActiveQuery's
condition number ~3e14, essentially singular) because they greedily chase
whichever pair looks most informative under the CURRENT (possibly still
wrong) theta_hat -- concentrating queries along one direction and leaving
others completely unprobed.

D-optimal design is the classical fix for exactly this failure mode in
experimental design: pick the next query to maximize det(V + X X^T), i.e.
the one that most increases the design matrix's determinant (its
"information volume"), with NO dependence on the current theta_hat at
all -- pure coverage, no exploitation. By the matrix determinant lemma,
det(V + XX^T) = det(V)(1 + X^T V^-1 X), so this reduces to maximizing
X^T V^-1 X -- exactly DPB's exploration-bonus term alone, with the
|X^T theta_hat| exploitation term removed entirely.
"""
import numpy as np

from src.preference.active_query import fit_logit_mle


class DOptimalStrategy:
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
        V = self._V(dim)
        V_inv = np.linalg.inv(V)

        best_score, best_pair = -np.inf, (0, 1)
        for i in range(K):
            for j in range(i + 1, K):
                X = feats[i] - feats[j]
                score = float(X @ V_inv @ X)  # maximize det(V + XX^T) increase
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
