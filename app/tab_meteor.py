"""Tab 1: Meteor live. Key exchange, encode a typed secret, per-step inspector, right vs wrong key.

Layout only: the encoding is stegaudit.meteor.encode (via appcore.meteor_demo, which records its
trace); demo mode reads the same dict from results/app_demo_meteor.json.
"""

from __future__ import annotations

import html
import json

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from stegaudit.appcore import DEMO_BETAS, DEMO_PROMPT, DEMO_SECRET, MAX_SECRET_CHARS, meteor_demo, step_detail
from stegaudit.provenance import REPO_ROOT
from style import BLUE, GRID, INK, INK2, ORANGE, style

DEMO_PATH = REPO_ROOT / "results" / "app_demo_meteor.json"


@st.cache_resource(show_spinner="Loading Qwen2.5-0.5B on the CPU ...")
def local_lm():
    from stegaudit.lm import LocalLM
    return LocalLM()


@st.cache_data(show_spinner="Encoding and decoding on the CPU ...")
def live_run(secret: str, beta: int) -> dict:
    return meteor_demo(local_lm(), secret, beta)


@st.cache_data
def demo_runs() -> dict | None:
    if not DEMO_PATH.exists():
        return None
    return json.loads(DEMO_PATH.read_text())["results"]["runs"]


def highlighted_text(r: dict, selected: int | None) -> str:
    """Payload tokens on an orange background, the inspected one outlined; cover tokens plain."""
    spans = []
    for k, t in enumerate(r["token_texts"]):
        t = html.escape(t).replace("\n", "<br>")
        if k < r["n_payload_tokens"]:
            border = "outline:2px solid #0b0b0b;" if k == selected else ""
            spans.append(f'<span title="payload step {k}" style="background:#fbd3c1;border-radius:3px;'
                         f'padding:0 1px;{border}">{t}</span>')
        else:
            spans.append(t)
    return ('<div style="font-size:0.98rem;line-height:1.8;padding:12px 14px;border:1px solid #e6e5e1;'
            'border-radius:8px;background:#fcfcfb">' + "".join(spans) + "</div>")


def step_figure(step: dict, beta: int):
    """Left: top-15 next-token probabilities (chosen one highlighted). Right: the intervals on [0, 2^β)."""
    probs, toks, pos = np.array(step["probs"]), step["tokens"], step["pos"]
    top = np.argsort(-probs, kind="stable")[:15]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.2), gridspec_kw={"width_ratios": [1.1, 1]})
    cols = [ORANGE if i == pos else BLUE for i in top]
    a1.barh(range(len(top)), probs[top], color=cols, height=0.7)
    a1.set_yticks(range(len(top)), [repr(toks[i]) for i in top], fontsize=7.5)
    a1.invert_yaxis()
    a1.set_xlabel("probability (top-k, renormalised)", fontsize=8, color=INK)
    a1.set_title(f"Next-token distribution, step {step['step']}", fontsize=9, color=INK, loc="left")
    style(a1)

    total = 2 ** beta
    lo, w = np.array(step["lo"], float) / total, np.array(step["widths"], float) / total
    for i in range(len(lo)):
        if w[i] > 0:
            a2.barh(0, w[i], left=lo[i], height=0.5, color=ORANGE if i == pos else "#c8d9f0",
                    edgecolor="white", linewidth=0.6)
    x = step["x"] / total
    a2.axvline(x, color=INK, linewidth=1.5)
    a2.text(x, 0.36, " x", fontsize=9, color=INK, va="bottom")
    a2.set_xlim(0, 1)
    a2.set_ylim(-0.5, 0.6)
    a2.set_yticks([])
    a2.set_xlabel("position in [0, 2^β), as a fraction", fontsize=8, color=INK)
    a2.set_title("Quantised intervals (chosen = orange)", fontsize=9, color=INK, loc="left")
    style(a2)
    a2.grid(axis="y", visible=False)
    a2.grid(axis="x", color=GRID)
    fig.tight_layout()
    return fig


def bits_table(step: dict, beta: int) -> str:
    """Monospace walk-through: chunk XOR mask = x; lo and hi-1 share a prefix; that prefix is sent."""
    d = step_detail(step, beta)
    n = d["n_bits"]

    def code(s, bold_prefix=False):                     # bold the first n bits (the common prefix)
        s = f"<b style='color:#c4441a'>{s[:n]}</b>{s[n:]}" if bold_prefix and n else s
        return f"<code style='font-size:0.9rem'>{s}</code>"
    rows = [("message chunk (next β bits)", code(d["chunk"], True)),
            ("mask (PRG: key, step)", code(d["mask"])),
            ("x = chunk XOR mask", code(d["x"])),
            ("interval start lo", code(d["lo"], True)),
            ("interval end hi − 1", code(d["hi_minus_1"], True)),
            ("common prefix length n", f"<b>{n}</b> → message bits sent: {code(d['bits_sent'] or '—')}")]
    body = "".join(f"<tr><td style='padding:3px 12px 3px 0;color:#52514e'>{k}</td><td>{v}</td></tr>" for k, v in rows)
    return f"<table style='border-collapse:collapse'>{body}</table>"


def render(demo: bool) -> None:
    st.subheader("Meteor: hiding a secret in a normal-looking study tip")
    st.write("Sender and Receiver agree on a key by X25519 key exchange. The Sender writes a study tip with the "
             "local model; at each token, the next β message bits XOR a keyed mask pick the token. Without the "
             "key, the choices look like ordinary random sampling.")

    if demo:
        runs = demo_runs()
        if runs is None:
            st.warning("No saved example yet (run experiments/app_demo_data.py), or switch demo mode off.")
            return
        c1, c2 = st.columns([1, 2])
        c1.text_input("Secret (demo mode: fixed example)", DEMO_SECRET, disabled=True, key="m_secret_demo")
        beta = c2.select_slider("Precision β", options=DEMO_BETAS, value=32, key="m_beta_demo")
        r = runs[str(beta)]
    else:
        c1, c2 = st.columns([1, 2])
        secret = c1.text_input(f"Secret (1–{MAX_SECRET_CHARS} characters)", DEMO_SECRET, max_chars=MAX_SECRET_CHARS,
                               key="m_secret")
        beta = c2.select_slider("Precision β", options=DEMO_BETAS, value=32, key="m_beta")
        if not secret:
            st.info("Type a secret.")
            return
        r = live_run(secret, beta)

    kx = r["key_exchange"]
    st.markdown("##### 1 · Key exchange (X25519 → HKDF-SHA256)")
    k = st.columns(3)
    k[0].markdown(f"**Sender** sends public key  \n`{kx['sender_public'][:16]}…`  \nderived key "
                  f"fingerprint `{kx['sender_fingerprint']}`")
    k[1].markdown(f"**Receiver** sends public key  \n`{kx['receiver_public'][:16]}…`  \nderived key "
                  f"fingerprint `{kx['receiver_fingerprint']}` "
                  + ("✅ same" if kx["sender_fingerprint"] == kx["receiver_fingerprint"] else "❌ differs"))
    k[2].markdown(f"**Eve** (own key pair) sees both public keys  \nher key fingerprint `{kx['eve_fingerprint']}` "
                  "❌ differs")

    st.markdown(f"##### 2 · The tip (prompt: “{r['prompt']}”)")
    nbits = len(r["message_bits"])
    st.caption(f"Secret “{r['secret']}” = {nbits} bits + {r['tag_bits']}-bit tag = {r['payload_bits']} payload bits, "
               f"sent in the first {r['n_payload_tokens']} tokens (orange); the rest is ordinary cover text.")
    labels = {s["step"]: f"{s['step']} · {r['token_texts'][s['step']]!r} · {s['n_bits']} b" for s in r["steps"]}
    sel = st.pills("Inspect a payload token (step · token · bits sent)", list(labels), format_func=labels.get,
                   default=next((s["step"] for s in r["steps"] if s["n_bits"] > 0), 0), key=f"m_pick_{beta}")
    st.markdown(highlighted_text(r, sel), unsafe_allow_html=True)

    if sel is not None:
        step = r["steps"][sel]
        st.markdown(f"##### 3 · Step {sel}: how token {r['token_texts'][sel]!r} carries "
                    f"{step['n_bits']} bit{'s' if step['n_bits'] != 1 else ''}")
        st.pyplot(step_figure(step, beta))
        plt.close("all")
        st.markdown(bits_table(step, beta), unsafe_allow_html=True)
        st.caption("Every x inside the chosen interval starts with the same n bits (red), so the token pins down "
                   "exactly those n bits of x. The Receiver rebuilds the same interval and XORs with the same mask "
                   "to recover them. n = 0 means this token carries nothing and the same bits are tried again.")

    st.markdown("##### 4 · Decoding")
    d = st.columns(2)
    right, wrong = r["decode_right"], r["decode_wrong"]
    if right["ok"]:
        d[0].success(f"Receiver (right key): tag ✓ → secret “{right['secret']}”")
    else:
        d[0].error(f"Receiver (right key) failed: {right['failure']}")
    if wrong["ok"]:
        d[1].warning("Eve's key passed the tag check (probability ≈ 2^-tag_bits; happens by chance).")
    else:
        d[1].error(f"Eve (wrong key): {wrong['failure'] or 'rejected'} → no message. Her bits: "
                   f"`{''.join(map(str, wrong['payload_bits'][:r['payload_bits']]))}` (garbage)")
    if not demo:
        st.caption("Live run on the CPU; results are cached per (secret, β).")
