"""Direct Preference Optimization (Rafailov et al., NeurIPS 2023) as a
baseline estimator for the linear style preference.

The policy over a situation's candidates is pi_theta(k) ~ exp(theta . phi_k)
and the reference policy is the data prior p(code | situation). Since the
normalized "human" feature is phi_h = (log p - mu_h) / sd_h, the reference is
itself in the family: pi_ref = pi_theta_ref with theta_ref = sd_h e_h. The DPO
loss on answered pairs (winner w, loser l),
    L(theta) = -mean log sigmoid(beta [(log pi_theta(w) - log pi_ref(w))
                                       - (log pi_theta(l) - log pi_ref(l))])
             = -mean log sigmoid(beta (theta - theta_ref) . (phi_w - phi_l)),
is minimized by --steps of full-batch gradient descent from theta_ref (DPO's
usual start pi_theta = pi_ref), with no prior and no posterior: the
reference anchor and the finite number of steps are its only regularizers.
Compared with the Bradley-Terry/Laplace estimator of bayesian_query.py it
differs in (i) the anchor at the human prior instead of a N(0, I) prior,
(ii) a point estimate (no covariance), so on its own it can only ask random
questions.

DPOStrategy(select="random") -- random pairs, DPO estimate (the plain baseline)
DPOStrategy(select="eubo")   -- pairs chosen by EUBO on the Laplace posterior
                                of the same answers, DPO estimate (isolates the
                                estimator from the query choice)
Recommendation: argmax pi_theta = argmax theta . phi.
"""
import numpy as np

from src.preference.bayesian_query import EUBOStrategy, _sigmoid


def dpo_fit(diff_feats, theta_ref, beta=1.0, lr=0.5, steps=100):
    """diff_feats: (N, d) rows winner - loser. Returns theta."""
    X = np.asarray(diff_feats, dtype=np.float64).reshape(-1, len(theta_ref))
    delta = np.zeros(len(theta_ref))
    for _ in range(steps):
        p = _sigmoid(beta * X @ delta)
        delta += lr * beta * X.T @ (1 - p) / len(X)
    return theta_ref + delta


class DPOStrategy(EUBOStrategy):
    def __init__(self, theta_ref, select="random", beta=1.0, lr=0.5, steps=100, rng=None):
        super().__init__(rng=rng)
        self.theta_ref = np.asarray(theta_ref, dtype=np.float64)
        self.mode, self.beta, self.lr, self.steps = select, beta, lr, steps
        self.dpo_theta = None

    def select(self, feats):
        if self.mode == "eubo":
            return super().select(feats)
        i, j = self.rng.choice(len(feats), size=2, replace=False)
        return int(i), int(j)

    def update(self, feats, i, j, winner):
        super().update(feats, i, j, winner)  # Laplace posterior, used only to choose EUBO pairs
        self.dpo_theta = dpo_fit(np.array(self.diff_feats), self.theta_ref, self.beta, self.lr, self.steps)

    @property
    def theta_hat(self):
        return self.dpo_theta

    @theta_hat.setter
    def theta_hat(self, value):
        self._laplace_mean = value

    def _posterior(self, dim):
        if self._cov is None:
            return np.zeros(dim), self.prior_std**2 * np.eye(dim)
        return self._laplace_mean, self._cov
