"""Helpers for the Streamlit app (app/streamlit_app.py). No algorithm lives in the app file.

Idea in plain words
-------------------
Tab 4 (audit game) lets a viewer change budget, stake inequality (Gini), audit
detection power d, false-flag rate f and penalty P, and see:
  * how each auditor policy spreads its coverage over the stakes,
  * how much stake each colluder gets through undetected under each policy,
  * how many decisions a best-responding colluder attacks (ties go to the
    auditor, PREREG ed19351), and
  * how many audits a test needs to catch a uniform-eps colluder at (d, f)
    (the P3a definition, PREREG 7088dd5).
Everything is computed by simulate.py / audit_lp.py, the same code experiment
04 uses, so the app cannot disagree with the experiment.

Security assumption and what breaks
-----------------------------------
Same as simulate.py: colluders know the coverage and stakes; d, f, P are
fixed numbers; "suspicion" auditing = random auditing because the channel is
undetectable from transcripts. If a transcript detector worked, the
suspicion policy shown here would understate what an auditor can do.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from stegaudit.provenance import REPO_ROOT
from stegaudit.simulate import (attack_best_response, attack_top_k, attack_uniform, audits_to_detect,
                                count_ties, coverage, expected_undetected, stakes)

POLICIES = ["uniform", "suspicion", "stake_proportional", "m1", "m2"]
POLICY_LABELS = {"uniform": "Uniform", "suspicion": "Suspicion (= random)",
                 "stake_proportional": "Stake-proportional", "m1": "LP M1", "m2": "LP M2"}
EPS_VALUES = [0.01, 0.05]          # registered colluder rates for 04 (PREREG 7a7fa45)
P_MULTS = [0, 1, 10, 100]          # P = mult x mean(v), as in 04
T_DEFAULT = 1000


def colluder_labels(eps_values=EPS_VALUES) -> list[str]:
    return (["best response"] + [f"top-k (k = {e:g}·T)" for e in eps_values]
            + [f"uniform ε = {e:g}" for e in eps_values])


def audit_game(dist: str, gini_target: float, b_frac: float, d: float, p_mult: float,
               T: int = T_DEFAULT, seed: int = 0, eps_values=EPS_VALUES) -> dict:
    """One instance of the audit game: every policy against every colluder (closed form)."""
    v = stakes(dist, gini_target, T, np.random.default_rng(seed))
    P, B = p_mult * float(v.mean()), b_frac * T
    out = {"v": v, "P": P, "B": B, "total_stake": float(v.sum()), "policies": {}}
    for pol in POLICIES:
        c = coverage(pol, v, B, d, P)
        colluders = {"best response": attack_best_response(c, v, d, P)}
        for e in eps_values:
            colluders[f"top-k (k = {e:g}·T)"] = attack_top_k(v, int(round(e * T)))
        for e in eps_values:
            colluders[f"uniform ε = {e:g}"] = attack_uniform(T, e)
        out["policies"][pol] = {
            "c": c,
            "budget_used": float(c.sum()),
            "attacked": int(colluders["best response"].sum()),
            "ties": count_ties(c, v, d, P),
            "undetected": {k: expected_undetected(a, c, v, d) for k, a in colluders.items()},
        }
    return out


def deterrence_boundary(v, d: float, P: float) -> np.ndarray:
    """Coverage at which attacking decision i stops paying: c_i = v_i / (d (v_i + P)), capped at 1."""
    v = np.asarray(v, float)
    return np.minimum(1.0, v / (d * (v + P)))


def audits_needed(d: float, f: float, eps_values=EPS_VALUES) -> dict:
    """Audits to catch a uniform-eps colluder (P3a definition; level 0.05, power 0.8 when f > 0)."""
    return {e: float(audits_to_detect(e, d, fpr=f)) for e in eps_values}


def load_d_estimate(path: Path = REPO_ROOT / "results" / "d_estimate.json") -> list[dict]:
    """Phase 2's (d, f) pairs, one per Sentinel; [] if Phase 2 has not produced them yet."""
    if not Path(path).exists():
        return []
    return json.loads(Path(path).read_text())["results"]["pairs"]
