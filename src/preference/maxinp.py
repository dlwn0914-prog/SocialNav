"""MaxInP ("Maximum Informative Pair"), ported from Saha 2021, "Optimal
Algorithms for Stochastic Contextual Preference Bandits" (NeurIPS 2021) --
a linear dueling bandit with a proven O~(d sqrt(T)) regret bound, the
UCB-family entry in utils/unified_query_benchmark.py.

Unlike DPB (src/preference/dpb.py), its selection rule has no |X^T theta_hat|
exploitation term. Each round it:
  1. keeps only candidates not yet confidently beaten by any other one,
     C_t = {x : for all y, (x - y)^T theta_hat + beta_t ||x - y||_{V^-1} >= 0}
  2. asks the pair in C_t with the largest ||x - y||_{V^-1} -- the pair whose
     outcome the history pins down least.
So it is elimination + uncertainty sampling, which is why it fits our
best-arm-identification goal better than a cumulative-regret UCB rule.

beta_t: the paper's width carries a 1/kappa factor (kappa = min logistic
slope over the parameter ball), which makes the set C_t never shrink within
N<=10 queries; we use the common practical width without 1/kappa,
sqrt(2 log(1/delta) + d log(1 + t / (lam d))), scaled by beta_scale. The
paper's t0 random warm-up rounds are dropped (t0=0): with theta_hat=0 and
V=lam*I, round 1 already reduces to "ask the most distant pair".

Estimator: the paper's plain MLE needs those warm-up rounds -- after 1-2
(separable) answers the MLE runs to the norm bound, and C_t collapsed to
1-2 candidates after the very first query in our sanity check. We use the
lam-regularized MAP instead (same lam as V, i.e. prior N(0, I/lam)), the
estimator regularized-logistic-bandit confidence sets are stated for.
"""
import numpy as np

from src.preference.bayesian_query import laplace_posterior


class MaxInPStrategy:
    def __init__(self, lam=1.0, delta=0.1, beta_scale=1.0, rng=None):
        self.lam = lam
        self.delta = delta
        self.beta_scale = beta_scale
        self.rng = rng or np.random.default_rng()
        self.diff_feats, self.labels = [], []
        self.theta_hat = None

    def _V(self, dim):
        V = self.lam * np.eye(dim)
        for X in self.diff_feats:
            V += np.outer(X, X)
        return V

    def _beta(self, dim):
        t = len(self.diff_feats)
        return self.beta_scale * np.sqrt(2 * np.log(1 / self.delta) + dim * np.log(1 + t / (self.lam * dim)))

    def active_set(self, feats):
        feats = np.asarray(feats, dtype=np.float64)
        K, dim = feats.shape
        theta = self.theta_hat if self.theta_hat is not None else np.zeros(dim)
        V_inv = np.linalg.inv(self._V(dim))
        beta = self._beta(dim)
        diffs = feats[:, None, :] - feats[None, :, :]  # (K, K, d): x_i - x_j
        width = np.sqrt(np.maximum(np.einsum("ijd,de,ije->ij", diffs, V_inv, diffs), 0.0))
        ucb = diffs @ theta + beta * width
        np.fill_diagonal(ucb, np.inf)
        return np.flatnonzero((ucb >= 0).all(axis=1)), width

    def select(self, feats):
        active, width = self.active_set(feats)
        if len(active) < 2:
            # one survivor: compare it against its closest-call rival so the query is still informative
            others = np.setdiff1d(np.arange(len(feats)), active)
            i = int(active[0]) if len(active) else int(self.best(feats))
            return i, int(others[np.argmax(width[i, others])])
        sub = width[np.ix_(active, active)]
        a, b = np.unravel_index(np.argmax(sub), sub.shape)
        return int(active[a]), int(active[b])

    def update(self, feats, i, j, winner):
        diff = feats[i] - feats[j] if winner == i else feats[j] - feats[i]
        self.diff_feats.append(diff)
        self.labels.append(1.0)
        self.theta_hat, _ = laplace_posterior(np.array(self.diff_feats), feats.shape[1], prior_std=1 / np.sqrt(self.lam))

    def best(self, feats):
        if self.theta_hat is None:
            return int(self.rng.integers(len(feats)))
        return int(np.argmax(np.asarray(feats) @ self.theta_hat))

    def done(self):
        return False
