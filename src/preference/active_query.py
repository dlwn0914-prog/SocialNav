"""Active pairwise-comparison query selection, ported from UserAlign
(NeurIPS 2025, "Inference-Time Personalized Alignment with a Few User
Preference Queries", https://github.com/machine-teaching-group/neurips2025-useralign).

Every other script in this repo picks which two candidates to compare next
*at random* (RandomQueryStrategy here matches that exactly). UserAlign
instead picks the pair expected to be most informative: it fits a Bradley-Terry
weight vector theta_hat via norm-constrained MLE (fit_logit_mle), then, for
each candidate j other than the current best, solves a small convex program
that asks "what is the best theta *consistent with every past answer* (within
its confidence region) that could still make j beat the current best?" -- the
candidate with the largest such value is the "challenger" most likely to
actually be informative to ask about next. This also gives a principled early
stop (done()): once no candidate could plausibly beat the current best even
under its most favorable within-confidence theta, further queries can't
change the outcome.

This is a from-scratch reimplementation (not a dependency on the
neurips2025-useralign package) so it plugs directly into this repo's existing
feature/scoring conventions (src/preference/features.py,
src/preference/bradley_terry.py) -- e.g. reusing bradley_terry.score() to
evaluate the final selection.
"""
import cvxpy as cp
import numpy as np


def fit_logit_mle(diff_features, labels, norm_bound=5.0):
    """Norm-constrained Bradley-Terry MLE (UserAlign's fit_logit_mle), as an
    alternative to bradley_terry.fit_preference_weights's L2-regularized
    gradient descent -- both are reasonable estimators for the same model;
    this one matches UserAlign's confidence-region theory exactly, which
    ActiveQueryStrategy's challenger search below depends on.
    diff_features: (N, D) array, features(A) - features(B) per comparison.
    labels: (N,) array, 1.0 if A won, 0.0 if B won.
    Returns: theta_hat, (D,) array with ||theta_hat|| <= norm_bound.
    """
    X = np.asarray(diff_features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    d = X.shape[1]
    theta = cp.Variable(d)
    logits = X @ theta
    # negative log-likelihood of a Bradley-Terry / logistic model
    loss = cp.sum(cp.logistic(-cp.multiply(2 * y - 1, logits)))
    constraints = [cp.norm(theta, 2) <= norm_bound]
    prob = cp.Problem(cp.Minimize(loss), constraints)
    for solver in ("CLARABEL", "ECOS", "SCS"):
        try:
            prob.solve(solver=solver, verbose=False)
            if theta.value is not None:
                return np.asarray(theta.value, dtype=np.float64)
        except Exception:
            continue
    raise RuntimeError("fit_logit_mle: all cvxpy solvers failed")


class RandomQueryStrategy:
    """Baseline matching every other script in this repo: pick a uniformly
    random pair each round."""

    def __init__(self, rng=None):
        self.rng = rng or np.random.default_rng()
        self.diff_feats, self.labels = [], []
        self.theta_hat = None

    def select(self, feats):
        i, j = self.rng.choice(len(feats), size=2, replace=False)
        return int(i), int(j)

    def update(self, feats, i, j, winner):
        diff = feats[i] - feats[j] if winner == i else feats[j] - feats[i]
        self.diff_feats.append(diff)
        self.labels.append(1.0)
        if len(self.diff_feats) >= 1:
            try:
                self.theta_hat = fit_logit_mle(np.array(self.diff_feats), np.array(self.labels))
            except RuntimeError:
                pass

    def best(self, feats):
        if self.theta_hat is None:
            return int(self.rng.integers(len(feats)))
        return int(np.argmax(feats @ self.theta_hat))

    def done(self):
        return False  # never stops early -- always uses the full query budget


class ActiveQueryStrategy:
    """UserAlign's version-space / confidence-region challenger search (see
    module docstring). eps: stop once no challenger's best-case score gap
    exceeds this (UserAlign's default epsilon)."""

    def __init__(self, norm_bound=5.0, delta=0.1, eps=1e-3, rng=None):
        self.norm_bound = norm_bound
        self.delta = delta
        self.eps = eps
        self.rng = rng or np.random.default_rng()
        self.diff_feats, self.labels = [], []
        self.theta_hat = None
        self._stop = False

    def _beta_t(self, t, dim):
        # UserAlign's CustomConfidenceInterval.beta_t-equivalent: a standard
        # self-normalized-martingale-style confidence radius for logistic
        # bandits, growing with dim and shrinking confidence (delta), and
        # mildly with round count t.
        return float(dim * np.log((1 + t) / self.delta) + 1.0)

    def select(self, feats):
        feats = np.asarray(feats, dtype=np.float64)
        dim = feats.shape[1]
        if self.theta_hat is None:
            theta = self.rng.standard_normal(dim)
            theta /= np.linalg.norm(theta)
            theta *= self.norm_bound * self.rng.random()
            self.theta_hat = theta

        scores = feats @ self.theta_hat
        best_idx = int(np.argmax(scores))

        diff_mat = None
        nll_hat = None
        beta_t = self._beta_t(len(self.diff_feats), dim)
        if self.diff_feats:
            X = np.array(self.diff_feats)
            y = np.array(self.labels)
            diff_mat = X * (2 * y[:, None] - 1)  # rows all point toward "winner preferred"
            logits = X @ self.theta_hat
            nll_hat = float(np.sum(np.logaddexp(0, -np.where(y == 1, 1, -1) * logits)))

        best_val, challenger_idx = -np.inf, None
        for j in range(len(feats)):
            if j == best_idx:
                continue
            theta_var = cp.Variable(dim)
            phi_param = feats[j] - feats[best_idx]
            constraints = [cp.norm(theta_var, 2) <= self.norm_bound]
            if diff_mat is not None:
                # theta_var must stay "consistent enough" with every past
                # answer: its log-loss on the history can't exceed the best
                # fit's by more than the confidence radius beta_t.
                logits_var = diff_mat @ theta_var
                nll_var = cp.sum(cp.logistic(-logits_var))
                constraints.append(nll_var <= nll_hat + beta_t)
            prob = cp.Problem(cp.Maximize(theta_var @ phi_param), constraints)
            solved = False
            for solver in ("CLARABEL", "ECOS", "SCS"):
                try:
                    prob.solve(solver=solver, verbose=False)
                    if prob.value is not None:
                        solved = True
                        break
                except Exception:
                    continue
            val = prob.value if solved and prob.value is not None else (scores[j] - scores[best_idx])
            if val > best_val:
                best_val, challenger_idx = val, j

        if challenger_idx is None or best_val <= self.eps:
            self._stop = True
            challenger_idx = int(self.rng.integers(len(feats)))
        return best_idx, challenger_idx

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
        return self._stop
