"""Posterior-sampling query selection -- the Thompson-Sampling-family and
Bayesian simple-regret entries in utils/unified_query_benchmark.py.

Both share one Laplace-approximated posterior over theta for the same
Bradley-Terry model as the rest of this package:
  prior theta ~ N(0, prior_std^2 I), likelihood P(i > j) = sigmoid(theta . (phi_i - phi_j)).
The MAP is found by Newton's method (d <= 7, so this is exact enough and
pure numpy); the posterior covariance is the inverse Hessian at the MAP.
prior_std=1.0 is a generic default, NOT tuned to the benchmark's ||theta*||.

TopTwoTSStrategy -- top-two Thompson sampling (Russo 2016; Jourdan et al.
NeurIPS 2022) mapped onto dueling feedback: draw theta~ from the posterior,
its argmax is the leader; keep redrawing until a draw's argmax differs --
that is the challenger; ask leader vs challenger. Structurally the TS
analogue of ActiveQueryStrategy's best-vs-challenger query. Caveat: the
top-two guarantees are for absolute-reward BAI; we found no proof for
pairwise feedback (dueling TS regret bounds exist, e.g. FGTS.CDB, Li et al.
ICML 2024, but for cumulative regret).

EUBOStrategy -- Expected Utility of the Best Option (Astudillo et al.,
AISTATS 2023, "qEUBO"): ask the pair maximizing E_theta[max(u_i, u_j)].
It is one-step Bayes-optimal for simple regret under noise-free answers,
which is exactly our objective (pick one best candidate after N queries).
Recommendation: argmax of posterior-mean utility.
"""
import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def laplace_posterior(diff_feats, dim, prior_std=1.0, iters=50):
    """diff_feats: (N, d) rows oriented winner - loser. Returns (theta_map, cov)."""
    prior_prec = 1.0 / prior_std**2
    theta = np.zeros(dim)
    X = np.asarray(diff_feats, dtype=np.float64).reshape(-1, dim)
    for _ in range(iters):
        p = _sigmoid(X @ theta)
        grad = X.T @ (1 - p) - prior_prec * theta
        H = (X * (p * (1 - p))[:, None]).T @ X + prior_prec * np.eye(dim)
        step = np.linalg.solve(H, grad)
        theta = theta + step
        if np.linalg.norm(step) < 1e-8:
            break
    p = _sigmoid(X @ theta)
    H = (X * (p * (1 - p))[:, None]).T @ X + prior_prec * np.eye(dim)
    return theta, np.linalg.inv(H)


class _LaplaceBase:
    def __init__(self, prior_std=1.0, rng=None):
        self.prior_std = prior_std
        self.rng = rng or np.random.default_rng()
        self.diff_feats = []
        self.theta_hat = None
        self._cov = None

    def _posterior(self, dim):
        if self._cov is None:
            return np.zeros(dim), self.prior_std**2 * np.eye(dim)
        return self.theta_hat, self._cov

    def update(self, feats, i, j, winner):
        self.diff_feats.append(feats[i] - feats[j] if winner == i else feats[j] - feats[i])
        self.theta_hat, self._cov = laplace_posterior(np.array(self.diff_feats), feats.shape[1], self.prior_std)

    def best(self, feats):
        if self.theta_hat is None:
            return int(self.rng.integers(len(feats)))
        return int(np.argmax(np.asarray(feats) @ self.theta_hat))

    def done(self):
        return False


class TopTwoTSStrategy(_LaplaceBase):
    def __init__(self, prior_std=1.0, max_resample=100, rng=None):
        super().__init__(prior_std, rng)
        self.max_resample = max_resample

    def select(self, feats):
        feats = np.asarray(feats, dtype=np.float64)
        mean, cov = self._posterior(feats.shape[1])
        draws = self.rng.multivariate_normal(mean, cov, size=self.max_resample + 1)
        utils = draws @ feats.T  # (draws, K)
        leader = int(np.argmax(utils[0]))
        for u in utils[1:]:
            if int(np.argmax(u)) != leader:
                return leader, int(np.argmax(u))
        # posterior is confident in the leader: fall back to its runner-up under the first draw
        u = utils[0].copy()
        u[leader] = -np.inf
        return leader, int(np.argmax(u))


class EUBOStrategy(_LaplaceBase):
    def __init__(self, prior_std=1.0, n_samples=512, rng=None):
        super().__init__(prior_std, rng)
        self.n_samples = n_samples

    def select(self, feats):
        feats = np.asarray(feats, dtype=np.float64)
        mean, cov = self._posterior(feats.shape[1])
        utils = feats @ self.rng.multivariate_normal(mean, cov, size=self.n_samples).T  # (K, S)
        eubo = np.maximum(utils[:, None, :], utils[None, :, :]).mean(axis=2)
        np.fill_diagonal(eubo, -np.inf)
        i, j = np.unravel_index(np.argmax(eubo), eubo.shape)
        return int(i), int(j)
