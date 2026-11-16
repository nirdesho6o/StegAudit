"""Experiment 04 analysis: P3a-P3d from results/04_audit_lp.json (read-only; prints a summary for LOG.md).

Committed BEFORE the registered 04 run finished, so the evaluation was fixed before the results were seen.
Any later change to this file is reported in LOG.md as a deviation.

Readings used (PREREG 7088dd5, 7a7fa45, 56cbaf3; P3b and P3c judged on expected undetected value):
  P3a: the slopes and MC power checks stored by 04 (criterion in PREREG 7088dd5).
  P3b: auditor uniform or suspicion; top-k colluder (k = eps*T) vs the uniform-eps colluder at the matched eps.
       Reported: cells where top-k leaves more value undetected than uniform-eps; the top-k colluder's
       undetected value as a share of TOTAL stake per Gini ("most" read as > 50 %), and whether that share
       rises with Gini in each (dist, B/T, d, P) series; the share of the ATTACKED stake left undetected.
  P3c: best-response colluder. M1 and M2 each vs the better of uniform and stake-proportional, per cell.
       Gain = (better baseline - LP) undetected value, as a share of total stake; "grows with Gini" =
       strictly increasing over the four Gini values in a (dist, B/T, d, P) series.
  P3d: M2 coverage recomputed per cell; proportional coverage would give a constant c_i / v_i among
       decisions with 0 < c_i < 1. Reported: coefficient of variation of c_i / v_i there, and the share of c_i = 0.
Usage: python experiments/04_analyze.py [path]
"""

from __future__ import annotations

import importlib.util
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np

from stegaudit.provenance import REPO_ROOT
from stegaudit.simulate import coverage, stakes

spec = importlib.util.spec_from_file_location("exp04", REPO_ROOT / "experiments" / "04_audit_lp.py")
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)


def stake_vector(dist, g):
    return stakes(dist, g, e.T, np.random.default_rng(e.seed_for("stakes", dist, g)))


def main(path: Path):
    R = json.loads(path.read_text())
    meta, res = R["meta"], R["results"]
    rows = res["rows"]
    print("meta:", {k: meta.get(k) for k in ("git_commit", "git_dirty", "timestamp")})
    print("summary:", res["summary"])
    idx = {(r["dist"], r["gini_target"], r["budget_frac"], r["d"], r["P_mult"], r["policy"], r["colluder"]): r
           for r in rows}
    D = sorted({r["d"] for r in rows})
    G, F, PM = e.GINIS, e.BUDGET_FRACS, e.P_MULTS
    V = {(dist, g): stake_vector(dist, g) for dist, g in product(e.DISTS, G)}
    for (dist, g), v in V.items():                      # the stakes rebuilt here are the ones 04 used
        r = idx[(dist, g, F[0], D[0], PM[0], "uniform", "uniform_eps0.01")]
        assert abs(r["gini_sample"] - float(e.gini(v))) < 1e-12

    print("\nP3a")
    for c in res["p3a"]:
        print(f"  f={c['fpr']} d={c['d']} src={c['phase2_sentinel']} slope="
              f"{None if c['slope'] is None else round(c['slope'], 3)} met={c['p3a_met']} fit_eps={c['fit_eps']} "
              f"mc_power_ok={sum(c['power_ok'])}/{len(c['power_ok'])}")

    print("\nP3b (auditor uniform/suspicion; top-k vs matched uniform-eps colluder)")
    for pol, eps in product(("uniform", "suspicion"), e.EPS_VALUES):
        k = round(eps * e.T)
        more, mono, n, ns, most = 0, 0, 0, 0, 0
        att_share = []
        for dist, frac, d, pm in product(e.DISTS, F, D, PM):
            series = []
            for g in G:
                tk = idx[(dist, g, frac, d, pm, pol, f"top_k{k}")]
                un = idx[(dist, g, frac, d, pm, pol, f"uniform_eps{eps}")]
                more += tk["undetected"] > un["undetected"]
                most += tk["undetected_frac_total"] > 0.5
                n += 1
                series.append(tk["undetected_frac_total"])
                att_share.append(tk["undetected"] / np.sort(V[(dist, g)])[::-1][:k].sum())
            ns += 1
            mono += bool(np.all(np.diff(series) > 0))
        print(f"  {pol} k={k}: top-k > uniform-eps in {more}/{n}; top-k undetected > 50 % of total stake in "
              f"{most}/{n}; share rises with Gini in {mono}/{ns} series; share of attacked stake undetected "
              f"min {min(att_share):.3f}")
        for dist in e.DISTS:
            parts = []
            for g in G:
                t = [idx[(dist, g, fr, d, pm, pol, f"top_k{k}")]["undetected_frac_total"]
                     for fr, d, pm in product(F, D, PM)]
                u = [idx[(dist, g, fr, d, pm, pol, f"uniform_eps{eps}")]["undetected_frac_total"]
                     for fr, d, pm in product(F, D, PM)]
                parts.append(f"G{g}: top-k {min(t):.3f}-{max(t):.3f} / uniform {min(u):.3f}-{max(u):.3f}")
            print(f"     {dist}: " + "; ".join(parts))

    print("\nP3c (best-response colluder; LP vs better of uniform and stake-proportional)")
    for lp in ("m1", "m2"):
        for pm in PM:
            beats, n, mono, ns = 0, 0, 0, 0
            worst = (np.inf, None)
            for dist, frac, d in product(e.DISTS, F, D):
                gains = []
                for g in G:
                    x = {p: idx[(dist, g, frac, d, pm, p, "best_response")]["undetected"]
                         for p in ("uniform", "stake_proportional", lp)}
                    base = min(x["uniform"], x["stake_proportional"])
                    gain = (base - x[lp]) / float(V[(dist, g)].sum())
                    beats += gain > 1e-9
                    n += 1
                    gains.append(gain)
                    if gain < worst[0]:
                        worst = (gain, (dist, g, frac, d))
                ns += 1
                mono += bool(np.all(np.diff(gains) > 0))
            print(f"  {lp} P={pm:g}*mean(v): beats both in {beats}/{n}; gain rises with Gini in {mono}/{ns}; "
                  f"worst gain {worst[0]:+.4f} at {worst[1]}")
        for g in G:
            gs = []
            for dist, frac, d, pm in product(e.DISTS, F, D, PM):
                x = {p: idx[(dist, g, frac, d, pm, p, "best_response")]["undetected"]
                     for p in ("uniform", "stake_proportional", lp)}
                gs.append((min(x["uniform"], x["stake_proportional"]) - x[lp]) / float(V[(dist, g)].sum()))
            print(f"    {lp} Gini {g}: gain share of total stake median {np.median(gs):+.4f} "
                  f"min {np.min(gs):+.4f} max {np.max(gs):+.4f}")

    print("\nP3d (M2 coverage vs proportional)")
    cvs, zero = [], []
    for dist, g, frac, d, pm in product(e.DISTS, G, F, D, PM):
        v = V[(dist, g)]
        c = coverage("m2", v, frac * e.T, d, pm * float(v.mean()))
        inner = (c > 1e-9) & (c < 1 - 1e-9)
        r = c[inner] / v[inner]
        cvs.append(float(np.std(r) / np.mean(r)) if inner.sum() > 1 else float("nan"))
        zero.append(float(np.mean(c <= 1e-9)))
    cvs = np.array(cvs)
    ok = ~np.isnan(cvs)
    print(f"  {len(cvs)} cells ({(~ok).sum()} with < 2 interior decisions); CV of c/v over 0<c<1: median "
          f"{np.median(cvs[ok]):.3f}, min {cvs[ok].min():.3f}; cells with CV < 0.01: {(cvs[ok] < 0.01).sum()}; "
          f"share of decisions with c = 0: median {np.median(zero):.3f}")

    s = res["summary"]
    print(f"\nMC check: {s['mc_outside_ci95']}/{s['n_rows']} outside 95 % ({s['mc_outside_ci95_frac']:.3f}), "
          f"{s['mc_beyond_4se']} beyond 4 SE")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "results" / "04_audit_lp.json")
