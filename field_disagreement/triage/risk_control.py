"""Distribution-free risk control over the cases the system AUTO-DECIDES.

Why this module exists rather than a threshold on the conformal set size:
split conformal (`triage.conformal`) guarantees MARGINAL coverage,

    P(Y in C(X)) >= 1 - alpha,

over a fresh draw from the same distribution. It does NOT say anything about
the subset a selection rule keeps. "Auto-decide the singletons" is a selection
event, and

    P(Y != Y_hat | |C(X)| = 1) <= alpha

does not follow from it - the singletons are exactly the cases the score was
most confident about, so the conditional risk is usually far BELOW alpha, but
"usually below" is not a guarantee and the direction is not proved. Claiming
the conditional statement from the marginal one is the single most common
over-claim in applied conformal work and this project made it too.

What is actually available is SELECTIVE risk control (Geifman & El-Yaniv's
selection-with-guaranteed-risk; Angelopoulos et al.'s Learn-then-Test as the
general recipe): pick the threshold on a held-out calibration split by testing
a sequence of candidate thresholds against a binomial tail bound, and stop at
the first one that fails. Then

    P( selective risk on a fresh draw <= alpha ) >= 1 - delta

holds with the threshold treated as fixed, because the sequence is FIXED IN
ADVANCE and tested in order - a fixed-sequence procedure needs no multiplicity
correction (`calibrate`).

TWO risks on the SAME kept set need no correction either, and the reason is
worth stating because this module got it wrong first and paid for it in
coverage. At one candidate the null is a UNION,

    H0(c) = { R_err(c) > alpha }  OR  { R_pan(c) > beta },

and a certificate is issued only when BOTH components are rejected. That is an
intersection-union test (Berger 1982): testing each component at level delta
gives a procedure of size <= delta, with no split across the risks. Bonferroni
here is not conservative-but-safe, it is answering a question nobody asked -
the union-null direction, where you would need a correction, is the one where
passing EITHER test suffices, and nothing in this module ever does that.
Splitting delta cost roughly 0.01 of certified panel risk at n = 150, which on
a cohort whose contested base rate is 0.194 is the difference between a
feasible certificate and none (`dual_calibrate`).

**The second risk is the contribution.** Nothing in the machinery cares what
the loss is, so the same certificate can be issued over a loss the panel
defines rather than the label:

    L_err(x)  = 1[ the auto-decision is wrong ]          model risk
    L_pan(x)  = 1[ d(x) > d0 ]                           panel risk: the panel
                                                         would have been split

Controlling both at once says something a conformal wrapper cannot: *every case
this system cleared without a human was one it was probably right about AND one
the panel would probably have agreed on*. A confidence score ranks the first
well and the second at chance, so at equal (alpha, beta) it has to defer more -
which is the comparison `dual_calibrate` is built to make.

Orientation: every function here takes a DEFER SCORE (higher = refer to a
human), matching `triage.deferral`. Auto-decide is `score <= threshold`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import lgamma, log

import numpy as np

__all__ = ["binom_upper", "selective_risk", "Certificate", "calibrate",
           "dual_calibrate", "apply_certificate"]


# --------------------------------------------------------------------------- #
# the bound
# --------------------------------------------------------------------------- #
def _binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Bin(n, p), summed in log space."""
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 1.0 if k >= n else 0.0
    lp, lq = log(p), log(1.0 - p)
    total = 0.0
    for i in range(int(k) + 1):
        lc = lgamma(n + 1) - lgamma(i + 1) - lgamma(n - i + 1)
        total += np.exp(lc + i * lp + (n - i) * lq)
    return float(min(total, 1.0))


def binom_upper(k: int, n: int, delta: float = 0.05) -> float:
    """Clopper-Pearson upper confidence bound on a binomial rate: the largest p
    with P(X <= k | n, p) >= delta. Exact, not asymptotic, which matters here
    because the interesting thresholds keep only a few hundred items and the
    realised risk is a handful of errors - a normal approximation on 3/400 is
    not a bound.

    n == 0 returns 1.0: a rule that keeps nothing has no evidence, so the
    honest bound is the vacuous one."""
    if n <= 0:
        return 1.0
    k = int(min(max(k, 0), n))
    if k >= n:
        return 1.0
    lo, hi = float(k) / n, 1.0
    for _ in range(80):                       # bisection: ~1e-24 on [0, 1]
        mid = 0.5 * (lo + hi)
        if _binom_cdf(k, n, mid) >= delta:
            lo = mid
        else:
            hi = mid
    return float(hi)


def selective_risk(loss, score, threshold: float) -> tuple[float, float, int]:
    """(realised risk among auto-decided, coverage, n kept) at a threshold."""
    loss = np.asarray(loss, dtype=float).ravel()
    score = np.asarray(score, dtype=float).ravel()
    keep = score <= threshold
    n = int(keep.sum())
    if n == 0:
        return float("nan"), 0.0, 0
    return float(loss[keep].mean()), float(n / len(score)), n


# --------------------------------------------------------------------------- #
# the certificate
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Certificate:
    """What a calibrated router carries into deployment.

    `thresholds` maps a signal name to the value it is auto-decided below, so a
    one-risk certificate has one entry and the dual-risk router has two and is
    a conjunction: auto-decide only if EVERY signal is under its threshold.

    `bound` is the guaranteed level per risk - the number the deployment is
    allowed to quote - and `realised` is what was measured on calibration,
    which is not a guarantee and must never be quoted as one. `feasible` False
    means no threshold in the sequence passed: the correct response is to defer
    everything, not to relax the bound after seeing the data."""

    thresholds: dict = field(default_factory=dict)
    alpha: dict = field(default_factory=dict)
    delta: float = 0.05
    bound: dict = field(default_factory=dict)
    realised: dict = field(default_factory=dict)
    coverage: float = 0.0
    n_cal: int = 0
    feasible: bool = False

    def to_dict(self) -> dict:
        return {"thresholds": dict(self.thresholds), "alpha": dict(self.alpha),
                "delta": self.delta, "bound": dict(self.bound),
                "realised": dict(self.realised), "coverage": self.coverage,
                "n_cal": self.n_cal, "feasible": self.feasible}

    @classmethod
    def from_dict(cls, block: dict | None) -> "Certificate":
        b = dict(block or {})
        return cls(thresholds=dict(b.get("thresholds", {})),
                   alpha=dict(b.get("alpha", {})), delta=float(b.get("delta", 0.05)),
                   bound=dict(b.get("bound", {})), realised=dict(b.get("realised", {})),
                   coverage=float(b.get("coverage", 0.0)),
                   n_cal=int(b.get("n_cal", 0)),
                   feasible=bool(b.get("feasible", False)))


def min_certifiable_n(alpha: float, delta: float) -> int:
    """Fewest auto-decided cases at which `alpha` is reachable AT ALL.

    With k = 0 errors the Clopper-Pearson bound is 1 - delta^(1/n), so below

        n* = log(delta) / log(1 - alpha)

    even a perfect record cannot certify the level - 59 cases for
    alpha = delta = 0.05. Candidate thresholds keeping fewer are skipped rather
    than tested, because a fixed-sequence procedure stops at its first failure
    and an unreachable first rung would abort the search before it began. The
    skip depends only on (alpha, delta, n), never on the losses, so it cannot
    leak the calibration data into the choice of threshold."""
    if not 0.0 < alpha < 1.0 or not 0.0 < delta < 1.0:
        raise ValueError("alpha and delta are rates in (0, 1)")
    return int(np.ceil(log(delta) / log(1.0 - alpha)))


def _candidates(score, n_grid: int) -> np.ndarray:
    """The fixed sequence of thresholds, most conservative first.

    Quantiles of the calibration scores, so the sequence is about COVERAGE
    (10%, 20%, ...) and not about the arbitrary scale of whichever signal is
    passed - a probability, a hinge score and an entropy are all admissible."""
    s = np.asarray(score, dtype=float).ravel()
    qs = np.linspace(1.0 / n_grid, 1.0, n_grid)
    return np.unique(np.quantile(s, qs))


def calibrate(score, loss, alpha: float = 0.05, delta: float = 0.05,
              n_grid: int = 100, name: str = "error") -> Certificate:
    """Largest auto-decision set whose selective risk is certified <= `alpha`.

    Fixed-sequence testing: candidate thresholds are ordered most-conservative
    first and each is tested against the Clopper-Pearson upper bound on the
    risk among the cases it keeps. The LAST threshold that passes before the
    first failure is returned, so the family-wise error is delta without any
    correction - that is what fixed-sequence testing buys, and it is why the
    sequence must not be re-ordered after looking at the data.

    Call it on a split that trained nothing and fitted nothing else - the same
    validation split the temperature and `tau_d` came from is NOT clean for
    this, because those were fitted to it."""
    s = np.asarray(score, dtype=float).ravel()
    ell = np.asarray(loss, dtype=float).ravel()
    if s.size != ell.size:
        raise ValueError(f"score/loss length mismatch: {s.size} vs {ell.size}")
    best = Certificate(alpha={name: float(alpha)}, delta=float(delta),
                       n_cal=int(s.size))
    n_min = min_certifiable_n(alpha, delta)
    for t in _candidates(s, n_grid):
        risk, cov, n = selective_risk(ell, s, t)
        if n < n_min:
            continue                                 # unreachable by size alone
        ub = binom_upper(int(round(risk * n)), n, delta)
        if ub > alpha:
            break                                   # stop at the first failure
        best = Certificate(thresholds={name: float(t)},
                           alpha={name: float(alpha)}, delta=float(delta),
                           bound={name: float(ub)}, realised={name: float(risk)},
                           coverage=float(cov), n_cal=int(s.size), feasible=True)
    return best


def dual_calibrate(scores: dict, losses: dict, alpha: dict,
                   delta: float = 0.05, n_grid: int = 20,
                   grids: dict | None = None) -> Certificate:
    """Maximise coverage subject to a certified bound on EVERY named risk.

        scores = {"error": r_err(x), "panel": r_pan(x)}
        losses = {"error": 1[wrong], "panel": 1[d > d0]}
        alpha  = {"error": 0.05,     "panel": 0.10}

    Auto-decide is the conjunction: every signal under its own threshold.

    THE PROCEDURE (Learn-then-Test, Angelopoulos et al. 2021). Every candidate
    threshold pair is a null hypothesis H(c) = "some risk at c exceeds its
    alpha", and a pair may be deployed only if H(c) is rejected by a procedure
    whose family-wise error over ALL the pairs it looked at is <= delta. The
    first risk is the CHAIN signal (`names[0]`, the error score), the others
    are held fixed per chain:

      - one chain per grid level of the non-chain signal(s), so |grid| chains;
      - within a chain the chain signal's threshold is walked in a fixed order,
        most conservative first, and the walk STOPS at the first pair that does
        not pass - a fixed-sequence test, level delta_chain with no correction;
      - the chains are combined by Bonferroni, delta_chain = delta / n_chains;
      - the certificate is the largest-coverage pair among ALL rejected nulls.
        Choosing among rejected hypotheses is free once FWER is controlled,
        which is the step the previous implementation lacked: it evaluated
        every pair at the full delta and kept the best feasible one, which is
        adaptive selection over ~|grid|^2 tests with no correction.

    Per pair the risks form an intersection-union test (Berger 1982): the pair
    passes only when EVERY risk's bound is under its alpha, and each is tested
    at the chain's full level - no split ACROSS the risks (module docstring).

    A structural property the comparison relies on survives: the router that
    carries the panel constraint on its own signal contains, as its loosest
    chain, the error-only walk on the confidence score, so at an equal
    (alpha, beta, delta) it cannot certify less than a confidence-only router.

    Residual caveat, stated rather than hidden: the candidate thresholds are
    quantiles of the CALIBRATION scores (losses are never consulted). Strict
    LTT fixes the grid before the data are seen; pass `grids` built on an
    independent split for the strict version."""
    names = list(scores)
    if set(names) != set(losses) or set(names) != set(alpha):
        raise ValueError("scores, losses and alpha must name the same risks")
    S = {k: np.asarray(scores[k], dtype=float).ravel() for k in names}
    L = {k: np.asarray(losses[k], dtype=float).ravel() for k in names}
    n = len(S[names[0]])
    if any(len(v) != n for v in (*S.values(), *L.values())):
        raise ValueError("every score and loss must have one entry per case")
    if grids is None:
        grids = {k: _candidates(S[k], n_grid) for k in names}
    grids = {k: np.sort(np.asarray(grids[k], dtype=float).ravel()) for k in names}

    chain, fixed = names[0], names[1:]
    chains = list(np.ndindex(*[len(grids[k]) for k in fixed])) if fixed else [()]
    d_chain = float(delta) / len(chains)      # Bonferroni across the chains
    n_min = max(min_certifiable_n(alpha[k], d_chain) for k in names)

    best = Certificate(alpha={k: float(alpha[k]) for k in names},
                       delta=float(delta), n_cal=int(n))
    for idx in chains:
        base = np.ones(n, dtype=bool)
        fixed_t = {}
        for k, i in zip(fixed, idx):
            fixed_t[k] = float(grids[k][i])
            base &= S[k] <= fixed_t[k]
        for t in grids[chain]:                # most conservative first
            keep = base & (S[chain] <= t)
            kept = int(keep.sum())
            if kept < n_min:
                continue                      # unreachable by size alone
            bound, realised, ok = {}, {}, True
            for k in names:
                ub = binom_upper(int(L[k][keep].sum()), kept, d_chain)
                bound[k], realised[k] = float(ub), float(L[k][keep].mean())
                ok = ok and ub <= alpha[k]
            if not ok:
                break                         # fixed sequence: stop this chain
            if kept / n > best.coverage:
                best = Certificate(thresholds={chain: float(t), **fixed_t},
                                   alpha={k: float(alpha[k]) for k in names},
                                   delta=float(delta), bound=bound,
                                   realised=realised, coverage=float(kept / n),
                                   n_cal=int(n), feasible=True)
    return best


DEFAULT_BETAS = (0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30)


def beta_frontier(error_score, panel_score, err_loss, pan_loss, *,
                  alpha: float = 0.05, betas=DEFAULT_BETAS,
                  delta: float = 0.05, n_grid: int = 20,
                  simultaneous: bool = False) -> list[dict]:
    """Certified coverage as a function of the panel constraint, not at one beta.

    A single declared `beta` turns the headline result into a coin-flip on a
    constant nobody can justify from outside the cohort. 0.10 sounds strict
    until you notice LIDC's contested base rate is 0.194, so it demands the
    router PURIFY the auto-decided set roughly fourfold before the binomial
    bound is even consulted; at that point every router reports coverage 0 and
    the table says nothing about any of them. Sweeping beta says what the single
    number was trying to: over the whole range where automation is possible at
    all, does routing on predicted panel dispersion certify more of it than
    routing on confidence?

    The curve also cannot be gamed. A reader who suspects beta was chosen to
    make the method win can read the coverage at every other beta in the same
    table, which is the property a declared constant does not have.

    WHAT EACH ROW GUARANTEES. With `simultaneous=False` (the default) every row
    is a certificate at level delta FOR THAT BETA ALONE. Reading one row off the
    frontier after looking at the calibration results (as `headline_beta`
    does) is a selection, and the selected row is then only covered at
    delta * len(betas) by the union bound. So the default frontier is a
    DESCRIPTIVE sensitivity analysis; quote a single post-selected beta as a
    guarantee only from a frontier built with `simultaneous=True`, which tests
    every beta at delta / len(betas) so that all rows hold at once."""
    betas = tuple(betas)
    if simultaneous:
        delta = float(delta) / len(betas)
    rows = []
    for b in betas:
        cert = dual_calibrate({"error": error_score, "panel": panel_score},
                              {"error": err_loss, "panel": pan_loss},
                              {"error": float(alpha), "panel": float(b)},
                              delta=delta, n_grid=n_grid)
        rows.append({"beta": float(b),
                     "feasible": int(bool(cert.feasible)),
                     "coverage": float(cert.coverage),
                     "bound_error": float(cert.bound.get("error", float("nan"))),
                     "bound_panel": float(cert.bound.get("panel", float("nan")))})
    return rows


def headline_beta(baseline_rows: list[dict]) -> float | None:
    """The beta the two routers are COMPARED at, fixed by a rule, not by eye.

    The rule is: the smallest beta on the grid at which the BASELINE router -
    confidence carrying both constraints - is feasible. It is deliberately the
    baseline that picks the point. Anywhere below it neither system automates
    anything and the comparison is empty; anywhere above it the constraint is
    slack and both drift toward the same coverage. The one place the contest is
    real is the tightest panel constraint the baseline can still meet, and
    naming it this way means the operating point cannot have been chosen after
    seeing where our own router happened to look best.

    Returns None when the baseline is feasible nowhere on the grid, which is
    itself the result: the panel constraint is then unreachable without a
    panel-aware signal, and the honest report is the frontier, not a number.

    The rule protects the COMPARISON from being chosen by our router, but the
    point is still chosen from calibration outcomes, so the certificate read
    there is covered at delta only when the frontier was built with
    `beta_frontier(..., simultaneous=True)`. Otherwise report it as an
    empirical operating point on a descriptive frontier, not as certified."""
    feas = [r["beta"] for r in baseline_rows
            if r["feasible"] and r["coverage"] > 0.0]
    return float(min(feas)) if feas else None


def apply_certificate(cert: Certificate, scores: dict) -> np.ndarray:
    """Boolean AUTO-DECIDE mask on fresh items: every signal under threshold.

    An infeasible certificate defers everything, which is the safe reading of
    "no threshold passed" and keeps the deployed rule honest rather than
    silently falling back to a coverage target."""
    first = np.asarray(scores[next(iter(scores))], dtype=float).ravel()
    if not cert.feasible:
        return np.zeros(first.shape, dtype=bool)
    keep = np.ones(first.shape, dtype=bool)
    for k, t in cert.thresholds.items():
        keep &= np.asarray(scores[k], dtype=float).ravel() <= t
    return keep
