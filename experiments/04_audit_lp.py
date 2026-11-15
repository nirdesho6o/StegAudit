"""Experiment 04: auditor policies vs colluders on synthetic stakes (no LLM calls).

Grid (CLAUDE.md): stakes lognormal / Pareto with Gini in {0.2, 0.4, 0.6, 0.8}, T = 1000,
budget B/T in {1 %, 2 %, 5 %, 10 %}, detection power d in {0.5, 0.8, 0.95} (+ d_estimate
from Phase 2 via --d-estimate), penalty P in {0, mean(v), 10 mean(v)}.
Auditor policies: uniform, suspicion (= B random audits), stake_proportional, m1, m2.
Colluders (PREREG 7a7fa45): uniform rate eps in EPS_VALUES and top-k with matched k = eps*T
(equal attack counts isolate targeting), plus best response to the announced coverage.
For every cell x policy x colluder: closed-form expected undetected value (and as a
fraction of total stake), colluders' expected net gain, expected detections, and a
Monte Carlo check with MC_DRAWS draws.

P3a section (PREREG clarification 7088dd5): audits needed to detect a uniform-eps colluder
via a one-sided binomial test with false-flag rate f, slope of log n vs log eps, and a
Monte Carlo power check.

Outputs: results/04_audit_lp.json, results/figs/04_*.png
Usage:   python experiments/04_audit_lp.py [--d-estimate X] [--fpr-phase2 F] [--quick]
"""

from __future__ import annotations

import argparse
import time
import zlib
from pathlib import Path

import numpy as np

from stegaudit.config import MeteorConfig
from stegaudit.provenance import REPO_ROOT, write_result_json
from stegaudit.simulate import (attack_best_response, attack_top_k, attack_uniform, audits_to_detect, coverage,
                                expected_colluder_gain, expected_detections, expected_undetected, gini, mc_power,
                                monte_carlo_undetected, stakes)

SEED = 0
T = 1000
DISTS = ["lognormal", "pareto"]
GINIS = [0.2, 0.4, 0.6, 0.8]
BUDGET_FRACS = [0.01, 0.02, 0.05, 0.10]
D_VALUES = [0.5, 0.8, 0.95]
P_MULTS = [0.0, 1.0, 10.0]                      # P = mult * mean(v)
POLICIES = ["uniform", "suspicion", "stake_proportional", "m1", "m2"]
EPS_VALUES = [0.01, 0.05]                       # uniform colluder rates (PREREG 7a7fa45)
# top-k colluders matched to each eps: k = eps * T (10 and 50 at T = 1000), so attack counts are equal
COLLUDERS = ([f"uniform_eps{e}" for e in EPS_VALUES] + [f"top_k{round(e * T)}" for e in EPS_VALUES]
             + ["best_response"])
MC_DRAWS = 10_000
P3A_EPS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
P3A_FPR = [0.0, 0.01, 0.05]
P3A_DELTA, P3A_POWER = 0.05, 0.8


def seed_for(*parts) -> int:
    return zlib.crc32("/".join(map(str, (SEED,) + parts)).encode())


def colluder_attacks(name: str, c, v, d: float, P: float) -> np.ndarray:
    """Attack probabilities a_i for a colluder named in COLLUDERS."""
    if name.startswith("uniform_eps"):
        return attack_uniform(T, float(name.removeprefix("uniform_eps")))
    if name.startswith("top_k"):
        return attack_top_k(v, int(name.removeprefix("top_k")))
    if name == "best_response":
        return attack_best_response(c, v, d, P)
    raise ValueError(name)


def run_grid(d_values, quick: bool):
    rows = []
    dists = DISTS[:1] if quick else DISTS
    ginis = GINIS[::2] if quick else GINIS
    fracs = BUDGET_FRACS[2:3] if quick else BUDGET_FRACS
    pm = P_MULTS[1:2] if quick else P_MULTS
    draws = 500 if quick else MC_DRAWS
    for dist in dists:
        for g in ginis:
            v = stakes(dist, g, T, np.random.default_rng(seed_for("stakes", dist, g)))
            for frac in fracs:
                B = frac * T
                for d in d_values:
                    for mult in pm:
                        P = mult * float(v.mean())
                        for pol in POLICIES:
                            c = coverage(pol, v, B, d, P)
                            for col in COLLUDERS:
                                a = colluder_attacks(col, c, v, d, P)
                                exact = expected_undetected(a, c, v, d)
                                mc = monte_carlo_undetected(
                                    a, c, v, d, draws,
                                    np.random.default_rng(seed_for("mc", dist, g, frac, d, mult, pol, col)),
                                    mode="subset" if pol == "suspicion" else "bernoulli", B=B)
                                z = (mc["mean"] - exact) / mc["se"] if mc["se"] > 0 else 0.0
                                rows.append({
                                    "dist": dist, "gini_target": g, "gini_sample": gini(v), "budget_frac": frac,
                                    "d": d, "P_mult": mult, "policy": pol, "colluder": col,
                                    "attacked": float(a.sum()),
                                    "undetected": exact, "undetected_frac_total": exact / float(v.sum()),
                                    "colluder_gain": expected_colluder_gain(a, c, v, d, P),
                                    "detections": expected_detections(a, c, d),
                                    "mc_mean": mc["mean"], "mc_se": mc["se"], "mc_z": z,
                                    "mc_in_ci95": abs(z) <= 1.96 or mc["se"] == 0,
                                })
            print(f"done {dist} gini={g} ({len(rows)} rows)", flush=True)
    return rows


def run_p3a(d_values):
    out = []
    rng = np.random.default_rng(seed_for("p3a"))
    for f in P3A_FPR:
        for d in d_values:
            ns = [audits_to_detect(e, d, P3A_DELTA, f, P3A_POWER) for e in P3A_EPS]
            powers = [mc_power(e, d, n, f, P3A_DELTA, 20_000, rng) for e, n in zip(P3A_EPS, ns)]
            regime = [e * d <= f for e in P3A_EPS] if f > 0 else [True] * len(P3A_EPS)
            x = np.log([e for e, r in zip(P3A_EPS, regime) if r])
            y = np.log([n for n, r in zip(ns, regime) if r])
            slope = float(np.polyfit(x, y, 1)[0]) if len(x) >= 2 else None
            target = 0.95 if f == 0 else P3A_POWER
            out.append({"fpr": f, "d": d, "eps": P3A_EPS, "n": ns, "mc_power": powers,
                        "power_target": target,
                        "power_ok": [abs(p - target) <= 0.03 for p in powers],
                        "fit_eps": [e for e, r in zip(P3A_EPS, regime) if r], "slope": slope,
                        "p3a_met": (slope is not None and -2.2 <= slope <= -1.8) if f > 0 else None})
            print(f"P3a f={f} d={d}: slope {slope}", flush=True)
    return out


def figures(rows, p3a, figdir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from stegaudit.simulate import coverage as cov

    ink, ink2, grid_c, surface = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
    series = [("uniform", "uniform / suspicion (random)", "#2a78d6"),
              ("stake_proportional", "stake-proportional", "#eb6834"),
              ("m1", "LP M1", "#1baf7a"), ("m2", "LP M2", "#eda100")]
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": ink2, "axes.labelcolor": ink, "xtick.color": ink2,
                         "ytick.color": ink2, "figure.facecolor": surface, "axes.facecolor": surface})
    figdir.mkdir(parents=True, exist_ok=True)

    def style(ax, title):
        ax.set_title(title, fontsize=9, color=ink)
        ax.grid(color=grid_c, linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    def pick(**kw):
        return [r for r in rows if all(r[k] == v for k, v in kw.items())]

    base = dict(colluder="best_response", d=0.8, P_mult=1.0)
    # 1) undetected share vs Gini (B/T = 5 %), one panel per stake distribution
    dists = sorted({r["dist"] for r in rows})
    fig, axes = plt.subplots(1, len(dists), figsize=(max(4.6 * len(dists), 7.0), 3.3), sharey=True)
    for ax, dist in zip(np.atleast_1d(axes), dists):
        for pol, lab, col in series:
            pts = sorted((r["gini_target"], r["undetected_frac_total"])
                         for r in pick(dist=dist, policy=pol, budget_frac=0.05, **base))
            if pts:
                ax.plot(*zip(*pts), marker="o", color=col, linewidth=2, markersize=5, label=lab)
        ax.set_xlabel("Gini of stakes")
        style(ax, f"{dist}: B/T = 5 %, d = 0.8, P = mean(v)")
    np.atleast_1d(axes)[0].set_ylabel("expected undetected stake / total stake")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Best-responding colluder: share of total stake that goes undetected", color=ink, fontsize=10)
    fig.tight_layout()
    fig.savefig(figdir / "04_undetected_vs_gini.png", dpi=150)
    plt.close(fig)

    # 2) undetected share vs budget (Gini 0.6)
    fig, axes = plt.subplots(1, len(dists), figsize=(max(4.6 * len(dists), 7.0), 3.3), sharey=True)
    for ax, dist in zip(np.atleast_1d(axes), dists):
        for pol, lab, col in series:
            pts = sorted((100 * r["budget_frac"], r["undetected_frac_total"])
                         for r in pick(dist=dist, policy=pol, gini_target=0.6, **base))
            if pts:
                ax.plot(*zip(*pts), marker="o", color=col, linewidth=2, markersize=5, label=lab)
        ax.set_xlabel("audit budget B/T (%)")
        style(ax, f"{dist}: Gini 0.6, d = 0.8, P = mean(v)")
    np.atleast_1d(axes)[0].set_ylabel("expected undetected stake / total stake")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir / "04_undetected_vs_budget.png", dpi=150)
    plt.close(fig)

    # 3) coverage profile c_i vs v_i for M1 and M2 (one instance)
    v = stakes("lognormal", 0.6, T, np.random.default_rng(seed_for("stakes", "lognormal", 0.6)))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3), sharey=True)
    for ax, (pol, lab, col) in zip(axes, (series[2], series[3])):
        c = cov(pol, v, 0.05 * T, 0.8, float(v.mean()))
        ax.scatter(v, c, s=8, color=col, alpha=0.6, linewidths=0)
        ax.set_xscale("log")
        ax.set_xlabel("stake v_i (log scale)")
        style(ax, f"{lab}: coverage per decision")
    axes[0].set_ylabel("coverage c_i")
    fig.suptitle("lognormal, Gini 0.6, B/T = 5 %, d = 0.8, P = mean(v)", color=ink, fontsize=10)
    fig.tight_layout()
    fig.savefig(figdir / "04_coverage_profile.png", dpi=150)
    plt.close(fig)

    # 4) P3a: audits needed vs eps (d = 0.8), one line per false-flag rate
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for (f, col) in zip(P3A_FPR, ("#2a78d6", "#eb6834", "#1baf7a")):
        r = next(x for x in p3a if x["fpr"] == f and x["d"] == 0.8)
        lab = f"f = {f}" + (f", slope {r['slope']:.2f}" if r["slope"] is not None else "")
        ax.plot(r["eps"], r["n"], marker="o", color=col, linewidth=2, markersize=5, label=lab)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("colluder rate ε")
    ax.set_ylabel("audits needed n(ε)")
    style(ax, "P3a: audits needed to detect a uniform-ε colluder (d = 0.8)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir / "04_p3a_audits_vs_eps.png", dpi=150)
    plt.close(fig)


def main(d_estimate, fpr_phase2, quick, out, figdir):
    global P3A_FPR
    d_values = D_VALUES + ([d_estimate] if d_estimate is not None else [])
    if fpr_phase2 is not None:
        P3A_FPR = P3A_FPR + [fpr_phase2]
    t0 = time.perf_counter()
    rows = run_grid(d_values, quick)
    p3a = run_p3a(d_values)
    outside95 = sum(not r["mc_in_ci95"] for r in rows)
    beyond4 = [r for r in rows if abs(r["mc_z"]) > 4]
    summary = {"n_rows": len(rows), "mc_outside_ci95": outside95,
               "mc_outside_ci95_frac": outside95 / len(rows), "mc_beyond_4se": len(beyond4),
               "d_estimate_included": d_estimate is not None, "quick": quick,
               "wall_s": round(time.perf_counter() - t0, 1)}
    print(summary, flush=True)
    write_result_json(out, {"summary": summary, "rows": rows, "p3a": p3a, "beyond_4se": beyond4},
                      MeteorConfig(), experiment="04_audit_lp", T=T, dists=DISTS, ginis=GINIS,
                      budget_fracs=BUDGET_FRACS, d_values=d_values, P_mults=P_MULTS, policies=POLICIES,
                      colluders=COLLUDERS, eps_values=EPS_VALUES, mc_draws=MC_DRAWS, seed=SEED,
                      p3a_eps=P3A_EPS, p3a_fpr=P3A_FPR, p3a_delta=P3A_DELTA, p3a_power=P3A_POWER)
    figures(rows, p3a, figdir)
    print(f"wrote {out} and figures in {figdir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-estimate", type=float, default=None)
    ap.add_argument("--fpr-phase2", type=float, default=None)
    ap.add_argument("--quick", action="store_true", help="tiny grid, 500 MC draws (smoke test)")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "04_audit_lp.json")
    ap.add_argument("--figdir", type=Path, default=REPO_ROOT / "results" / "figs")
    a = ap.parse_args()
    main(a.d_estimate, a.fpr_phase2, a.quick, a.out, a.figdir)
