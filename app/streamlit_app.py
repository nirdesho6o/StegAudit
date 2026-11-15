"""StegAudit app (Component B). Run:  streamlit run app/streamlit_app.py   (open a tab directly: ?tab=N)

Layout only: every number comes from stegaudit (appcore, meteor, simulate, audit_lp) or from saved
results in results/. One module per tab: tab_meteor, tab_detect, tab_gap, tab_audit.
Demo mode (default): no local model, no API calls; tabs read saved results.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import streamlit as st

import tab_audit
import tab_detect
import tab_gap
import tab_meteor

st.set_page_config(page_title="StegAudit", layout="wide")
st.title("StegAudit")
st.caption("Meteor steganography between colluding agents, and where an auditor should look.")
demo = st.sidebar.toggle("Demo mode (saved results only; no local model, no API calls)", value=True,
                         help="Off: tab 1 loads Qwen2.5-0.5B on the CPU (~10 s) and encodes live.")

TAB_NAMES = ["1 · Meteor channel", "2 · Detectability", "3 · Collusion and audit", "4 · Audit game"]
_q = st.query_params.get("tab", "1")                     # deep link, e.g. ?tab=4 (used for screenshots)
tabs = st.tabs(TAB_NAMES, default=TAB_NAMES[int(_q) - 1] if _q and _q in "1234" else None)
for tab, module in zip(tabs, (tab_meteor, tab_detect, tab_gap, tab_audit)):
    with tab:
        module.render(demo)
