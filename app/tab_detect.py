"""Tab 2: can a transcript monitor tell Meteor text from normal text? (saved results of 01, 02, 02b)."""

from __future__ import annotations

import html

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from stegaudit.appcore import detector_rows, text_pair
from style import BLUE, INK, INK2, ORANGE, style


@st.cache_data
def rows():
    return detector_rows()


@st.cache_data
def pair(i: int):
    return text_pair(i)


def auroc_figure(rs):
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(rs) + 1.1))
    for k, r in enumerate(rs):
        col = BLUE if r["view"] == "whole" else ORANGE
        ax.plot(r["ci95"], [k, k], color=col, linewidth=2.2)
        ax.plot(r["auroc"], k, "o", color=col, markersize=6)
    ax.axvline(0.5, color=INK2, linestyle="--", linewidth=1)
    ax.set_yticks(range(len(rs)), [f"{r['detector']} · {'whole text' if r['view'] == 'whole' else 'payload prefix'}"
                                   for r in rs], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("AUROC (Meteor vs normal), 95 % bootstrap CI · 0.5 = cannot tell apart", fontsize=8, color=INK)
    style(ax)
    fig.tight_layout()
    return fig


def render(demo: bool) -> None:
    st.subheader("Can a monitor spot the hidden message?")
    st.write("200 Meteor tips (β = 32) vs 200 normal tips from the same prompts, model and sampling settings. "
             "All numbers are saved results of experiments 01 and 02; nothing runs here.")

    st.markdown("##### 1 · Quiz: which one hides a message?")
    i = st.slider("Text pair", 0, 199, 3, key="d_pair")
    p = pair(i)
    texts = [("meteor", p["meteor_text"]), ("normal", p["normal_text"])]
    if not p["meteor_on_left"]:
        texts.reverse()
    reveal = st.toggle("Reveal", key=f"d_reveal_{i}")
    st.caption(f"Prompt: “{p['prompt']}”")
    cols = st.columns(2)
    for col, (kind, text), name in zip(cols, texts, ("Text A", "Text B")):
        with col:
            title = name if not reveal else f"{name}: {'Meteor (hidden message)' if kind == 'meteor' else 'normal sample'}"
            st.markdown(f"**{title}**")
            st.markdown(f'<div style="padding:10px 12px;border:1px solid #e6e5e1;border-radius:8px;'
                        f'background:#fcfcfb;min-height:150px">{html.escape(text).replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)
            if reveal:
                j = p["judge"][kind]
                st.caption(f"mean log-prob {p[kind + '_logprob']:.3f} · LLM judge P(hidden) "
                           f"{'—' if j is None else f'{j} / 100'}"
                           + (f" · payload in the first {p['n_payload_tokens']} tokens" if kind == "meteor" else ""))

    st.markdown("##### 2 · Every detector, 400 texts")
    rs = rows()
    st.pyplot(auroc_figure(rs))
    plt.close("all")
    st.caption("Blue: whole text (what a real monitor sees). Orange: only the payload prefix, which hands the "
               "monitor the payload boundary; the prefix log-prob signal (0.447) is a boundary artefact "
               "(exploratory 02b: last payload token alone 0.169, boundary-free 0.502). Vanishing-token and "
               "outside-top-k detectors found 0 events in both groups at β = 32, so their AUROC is exactly 0.5.")
    st.dataframe(pd.DataFrame([{"detector": r["detector"], "view": r["view"], "AUROC": round(r["auroc"], 3),
                                "95 % CI": f"[{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]",
                                "events (Meteor / normal)": "—" if r["events_meteor"] is None
                                else f"{r['events_meteor']} / {r['events_normal']}"} for r in rs]),
                 hide_index=True, width="stretch")
