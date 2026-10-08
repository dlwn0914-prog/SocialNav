"""Discounted Preference Bandits (DPB), ported from Kim et al. 2024,
"Time-Varying Preference Bandits for Robot Behavior Personalization"
(Applied Sciences 14(23):11002) -- a theoretically-grounded alternative to
src/preference/active_query.py's ActiveQueryStrategy (ported from UserAlign,
NeurIPS 2025), which has no proven regret bound.

Same pairwise-trajectory-preference setup as our existing code: a context
vector X := feat(A) - feat(B), a linear reward r(xi;theta) = theta . Phi(xi),
binary feedback R in {0,1} via a logistic model P(R=1|X,theta) = sigmoid(X.theta)
-- this is exactly our Bradley-Terry model (bradley_terry.py), just written
in the bandit paper's notation.

DPB's actual contribution over a plain (UserAlign-style) dueling bandit is
the discount factor gamma: past comparisons are down-weighted by
gamma**(t-s) in both the design matrix V_t and the MLE objective, so the
method can track a user whose TRUE preference drifts mid-session (not just
converge to a fixed one). gamma=1.0 degenerates to the stationary case.

Query selection (paper's Eq. 4): X_t = argmax_X |X^T theta_hat| + alpha * ||X||_{V^-1}
  - |X^T theta_hat|: how much the current estimate thinks this pair would
    discriminate (both X and -X carry the same information, hence the abs).
  - alpha * ||X||_{V^-1}: UCB-style exploration bonus, large for pairs whose
    direction the discounted history hasn't covered much.
"""
import cvxpy as cp
import numpy as np


def fit_discounted_mle(diff_features, labels, weights, norm_bound=5.0):
    """diff_features: (N, D) feat_a - feat_b per comparison, in the order
    they were asked. labels: (N,) 1.0 if a won. weights: (N,) gamma**(t-s)
    discount weight per comparison (most recent = 1.0). Penalized,
    discounted log-likelihood (paper's Eq. 5), solved via cvxpy."""
    X = np.asarray(diff_features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    theta = cp.Variable(X.shape[1])
    logits = X @ theta
    # weighted logistic negative log-likelihood, light L2 penalty (paper's lambda||theta||^2)
    nll = cp.sum(cp.multiply(w, cp.logistic(-cp.multiply(2 * y - 1, logits))))
    loss = nll + 0.5 * cp.sum_squares(theta)
    prob = cp.Problem(cp.Minimize(loss), [cp.norm(theta, 2) <= norm_bound])
    for solver in ("CLARABEL", "ECOS", "SCS"):
        try:
            prob.solve(solver=solver, verbose=False)
            if theta.value is not None:
                return np.asarray(theta.value, dtype=np.float64)
        except Exception:
            continue
    raise RuntimeError("fit_discounted_mle: all cvxpy solvers failed")


class DPBStrategy:
    """gamma=1.0: stationary case (no discounting -- degenerates to a
    plain UCB dueling bandit). gamma<1.0: tracks a drifting true
    preference, at the cost of some stationary-case efficiency (per the
    paper's own regret bound, which is looser the faster theta* drifts)."""

    def __init__(self, gamma=1.0, lam=1.0, norm_bound=5.0, alpha_scale=1.0, rng=None):
        self.gamma = gamma
        self.lam = lam
        self.norm_bound = norm_bound
        self.alpha_scale = alpha_scale
        self.rng = rng or np.random.default_rng()
        self.diff_feats, self.labels, self.weights_at_round = [], [], []
        self.t = 0
        self.theta_hat = None

    def _V(self, dim):
        V = self.lam * np.eye(dim)
        for s, X in enumerate(self.diff_feats):
            age = self.t - self.weights_at_round[s]
            V += (self.gamma ** age) * np.outer(X, X)
        return V

    def select(self, feats):
        feats = np.asarray(feats, dtype=np.float64)
        K, dim = feats.shape
        if self.theta_hat is None:
            self.theta_hat = np.zeros(dim)

        V = self._V(dim)
        V_inv = np.linalg.inv(V)
        alpha = self.alpha_scale * np.sqrt(dim * np.log(1 + self.t)) + 1.0

        best_score, best_pair = -np.inf, (0, 1)
        for i in range(K):
            for j in range(i + 1, K):
                X = feats[i] - feats[j]
                score = abs(float(X @ self.theta_hat)) + alpha * np.sqrt(max(X @ V_inv @ X, 0.0))
                if score > best_score:
                    best_score, best_pair = score, (i, j)
        return best_pair

    def update(self, feats, i, j, winner):
        X = feats[i] - feats[j]
        label = 1.0 if winner == i else 0.0
        self.diff_feats.append(X)
        self.labels.append(label)
        self.weights_at_round.append(self.t)
        self.t += 1

        dim = len(X)
        weights = np.array([self.gamma ** (self.t - 1 - r) for r in self.weights_at_round])
        try:
            self.theta_hat = fit_discounted_mle(
                np.array(self.diff_feats), np.array(self.labels), weights, norm_bound=self.norm_bound
            )
        except RuntimeError:
            pass

    def best(self, feats):
        if self.theta_hat is None:
            return int(self.rng.integers(len(feats)))
        return int(np.argmax(np.asarray(feats) @ self.theta_hat))

    def done(self):
        return False  # paper's method always uses the query budget; no early-stop rule given
