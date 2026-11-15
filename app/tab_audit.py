"""Tab 4: the audit game on synthetic stakes (layout only; numbers from stegaudit.appcore)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from stegaudit.appcore import (EPS_VALUES, P_MULTS, POLICIES, POLICY_LABELS, audit_game, audits_needed,
                               colluder_labels, deterrence_boundary, load_d_estimate)
from style import COLORS, INK, INK2, style

@st.cache_data(show_spinner="Solving the audit LPs ...")
def cached_game(dist, gini, b_frac, d, p_mult, seed):
    return audit_game(dist, gini, b_frac, d, p_mult, seed=seed)


def render(demo: bool) -> None:
    st.subheader("Audit game on synthetic stakes")
    st.write("T = 1000 decisions with random stakes. The auditor spreads a budget of B audits; an audit of a "
             "colluded decision detects with probability d; detection costs the colluders P. Colluders see the "
             "coverage and pick what to attack.")
    pairs = load_d_estimate()
    left, right = st.columns([1, 2.4], gap="large")
    with left:
        dist = st.radio("Stake distribution", ["lognormal", "pareto"], horizontal=True)
        gini = st.slider("Stake inequality (Gini)", 0.1, 0.9, 0.6, 0.05, key="a_gini")
        b_pct = st.slider("Budget B/T (%)", 0.5, 20.0, 2.0, 0.5)
        source = st.radio("d and f from", ["sliders"] + [f"Phase 2, Sentinel {p['sentinel']}" for p in pairs],
                          help="Phase 2 options appear once results/d_estimate.json exists.")
        if source == "sliders":
            d = st.slider("Detection power per audit d", 0.05, 1.0, 0.8, 0.05)
            f = st.slider("False-flag rate f (honest decision flagged)", 0.0, 0.3, 0.05, 0.01)
        else:
            p = pairs[[f"Phase 2, Sentinel {q['sentinel']}" for q in pairs].index(source)]
            d, f = float(p["d"]), float(p["f"])
            st.write(f"d = {d:.3f}, f = {f:.3f} (95 % CIs: d {np.round(p['d_ci95'], 3)}, f {np.round(p['f_ci95'], 3)})")
        p_mult = st.select_slider("Penalty P (× mean stake)", options=P_MULTS, value=10)
        policy = st.selectbox("Policy in detail", POLICIES, index=3, format_func=POLICY_LABELS.get, key="a_policy")
        colluder = st.selectbox("Colluder for the comparison", colluder_labels())
        seed = st.number_input("Stake seed", 0, 10_000, 0)

    g = cached_game(dist, gini, b_pct / 100, d, p_mult, int(seed))
    v, P = g["v"], g["P"]
    sel = g["policies"][policy]

    with right:
        m = st.columns(4)
        m[0].metric("Best response attacks", f"{sel['attacked']} / {len(v)}",
                    help="Attacks i only if gain > 1e-9·(v_i + P): ties go to the auditor (PREREG ed19351).")
        m[1].metric("Ties (indifferent)", sel["ties"])
        m[2].metric("Undetected stake", f"{100 * sel['undetected'][colluder] / g['total_stake']:.1f} %",
                    help=f"Expected undetected value vs '{colluder}', as % of total stake.")
        m[3].metric("Budget used", f"{sel['budget_used']:.1f} / {g['B']:.0f}")

        order = np.argsort(v)
        fig, ax = plt.subplots(figsize=(7.5, 3.4))
        for pol in POLICIES:
            on = pol == policy
            ax.plot(v[order], g["policies"][pol]["c"][order], color=COLORS[pol], linewidth=2.6 if on else 1.4,
                    alpha=1 if on else 0.55, label=POLICY_LABELS[pol], zorder=3 if on else 2)
        if d > 0:
            ax.plot(v[order], deterrence_boundary(v, d, P)[order], color=INK2, linestyle="--", linewidth=1,
                    label="deterrence boundary v/(d(v+P))")
        ax.set_xscale("log")
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel("stake v_i (log scale)", color=INK, fontsize=9)
        ax.set_ylabel("coverage c_i", color=INK, fontsize=9)
        ax.set_title("Coverage vs stake per policy", color=INK, fontsize=10, loc="left")
        style(ax)
        ax.legend(frameon=False, fontsize=7.5, loc="upper left")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Uniform and suspicion coincide by construction: against Meteor, transcript suspicion carries "
                   "no information, so suspicion auditing = random auditing (expected coverage B/T).")

        und ={pol: 100 * g["policies"][pol]["undetected"][colluder] / g["total_stake"] for pol in POLICIES}
        fig, ax = plt.subplots(figsize=(7.5, 2.6))
        ys = np.arange(len(POLICIES))
        ax.barh(ys, [und[p] for p in POLICIES], height=0.55, color=[COLORS[p] for p in POLICIES])
        for y, p in zip(ys, POLICIES):
            ax.text(und[p], y, f" {und[p]:.1f} %", va="center", fontsize=8, color=INK)
        ax.set_yticks(ys, [POLICY_LABELS[p] for p in POLICIES])
        ax.invert_yaxis()
        ax.set_xlim(0, max(1.0, max(und.values()) * 1.18))
        ax.set_xlabel("expected undetected value, % of total stake", color=INK, fontsize=9)
        ax.set_title(f"Undetected value vs colluder: {colluder}", color=INK, fontsize=10, loc="left")
        style(ax)
        ax.grid(axis="y", visible=False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        rows = []
        for pol in POLICIES:
            r = g["policies"][pol]
            rows.append({"policy": POLICY_LABELS[pol], "best-response attacks": r["attacked"], "ties": r["ties"],
                         **{f"{k} (% stake)": round(100 * u / g["total_stake"], 2) for k, u in r["undetected"].items()}})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

        n = audits_needed(d, f)
        st.write("**Audits needed to catch a uniform-ε colluder** at this d and f "
                 + ("(f = 0: P(at least one detection) ≥ 95 %)" if f == 0 else
                    "(one-sided binomial test, level 0.05, power 0.8)") + ": "
                 + ", ".join(f"ε = {e:g}: {np.ceil(n[e]):,.0f}" for e in EPS_VALUES))
        if demo:
            st.caption("Tab 4 computes live even in demo mode: it only solves LPs on synthetic stakes "
                       "(no local model, no API).")
