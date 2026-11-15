"""Experiment 01: Meteor round trip on the local Qwen model (design fixed in PREREG.md).

What it does
------------
* 200 messages = 20 study-tip prompts x 10 random 8-bit messages (+ 8-bit tag).
* Main run, beta = 32: full texts (payload, then cover text until EOS or
  120 tokens in total).
* beta sweep 8, 12, 16: payload phase only (max_new_tokens = 0), same
  (prompt, message) pairs.
* msg_id = 10 000 * run_index + i, so no (key, msg_id) pair is ever reused.
* For each message: encode, decode from TEXT (retokenised, like a real
  Receiver), and, only when retokenisation does not give back the Sender's
  ids, also decode from the ids. When retokenisation matches, the text path
  and the ids path receive identical input, so the text result IS the ids
  result. The fresh-process check (01b_redecode.py) recomputes both paths for
  every message.

Outputs
-------
results/01_meteor_roundtrip.json   per-message records + per-beta summary
results/figs/01_tokens_for_payload.png
results/figs/01_vanishing_and_capacity.png

Usage
-----
    python experiments/01_meteor_roundtrip.py                     # full run (~1 h on CPU)
    python experiments/01_meteor_roundtrip.py --limit 3 --out /tmp/x.json   # smoke test
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from stegaudit.config import MeteorConfig
from stegaudit.keys import pair_from_seed
from stegaudit.lm import LocalLM
from stegaudit.meteor import decode, decode_ids, encode
from stegaudit.provenance import REPO_ROOT, write_result_json

SEED = 0
SUBJECTS = ["chemistry", "history", "algebra", "biology", "geography", "physics",
            "literature", "economics", "music", "programming", "statistics", "philosophy",
            "art history", "astronomy", "psychology", "geometry", "spanish vocabulary",
            "world religions", "computer networks", "anatomy"]
MESSAGES_PER_PROMPT = 10
RUNS = [(32, 120), (8, 0), (12, 0), (16, 0)]        # (beta, max_new_tokens); index = run_index
MSG_ID_STRIDE = 10_000
KILL_RULE_MISMATCH_RATE = 0.05


def prompt_for(i: int) -> str:
    return f"Write a short study tip about {SUBJECTS[i // MESSAGES_PER_PROMPT]}."


def make_messages(n: int, message_bits: int) -> list[list[int]]:
    rng = np.random.default_rng(SEED)
    return [[int(b) for b in rng.integers(0, 2, message_bits)] for _ in range(n)]


def run_one(lm, keys_s, keys_r, cfg, run_index, i, msg) -> dict:
    p, msg_id = prompt_for(i), MSG_ID_STRIDE * run_index + i
    t0 = time.perf_counter()
    enc = encode(lm, keys_s, p, msg, msg_id, cfg)
    t_enc = time.perf_counter() - t0
    t0 = time.perf_counter()
    dec_t = decode(lm, keys_r, p, enc.text, msg_id, cfg)
    t_dec = time.perf_counter() - t0
    retok_ok = lm.text_to_ids(enc.text) == enc.ids
    dec_i = dec_t if retok_ok else decode_ids(lm, keys_r, p, enc.ids, msg_id, cfg)
    return {
        "i": i, "prompt": p, "msg_id": msg_id, "message": msg,
        "ids": enc.ids, "text": enc.text, "payload": enc.payload,
        "n_payload_tokens": enc.n_payload_tokens, "bits_per_token": enc.bits_per_token,
        "n_cover_tokens": enc.n_cover_tokens, "stop_reason": enc.stop_reason,
        "n_vanished_payload": enc.n_vanished_payload,
        "n_vanished_cover": enc.n_vanished - enc.n_vanished_payload,
        "eos_masked_steps": enc.eos_masked_steps, "eos_mass_removed": enc.eos_mass_removed,
        "retok_match": retok_ok,
        "ids_path_ok": dec_i.ok and dec_i.message_bits == msg,
        "ids_path_computed_separately": not retok_ok,
        "text_path_ok": dec_t.ok and dec_t.message_bits == msg,
        "text_path_failure": dec_t.failure,
        "text_path_wrong_accepted": dec_t.ok and dec_t.message_bits != msg,
        "encode_s": round(t_enc, 3), "decode_text_s": round(t_dec, 3),
    }


def summarise(recs: list[dict], beta: int, max_new_tokens: int) -> dict:
    n = len(recs)
    k_mis = sum(not r["retok_match"] for r in recs)
    ci = binomtest(k_mis, n).proportion_ci(confidence_level=0.95, method="exact") if n else None
    pay_tok = np.array([r["n_payload_tokens"] for r in recs])
    pay_steps = int(pay_tok.sum())
    cover_steps = sum(r["n_cover_tokens"] + (r["stop_reason"] == "eos") for r in recs)
    raw_bits = sum(sum(r["bits_per_token"]) for r in recs)
    return {
        "beta": beta, "max_new_tokens": max_new_tokens, "n_messages": n,
        "ids_path_ok": sum(r["ids_path_ok"] for r in recs),
        "text_path_ok": sum(r["text_path_ok"] for r in recs),
        "text_path_wrong_accepted": sum(r["text_path_wrong_accepted"] for r in recs),
        "text_path_failures": {f: sum(r["text_path_failure"] == f for r in recs)
                               for f in sorted({str(r["text_path_failure"]) for r in recs if r["text_path_failure"]})},
        "retok_mismatches": k_mis,
        "retok_mismatch_rate": k_mis / n if n else None,
        "retok_mismatch_ci95_exact": [float(ci.low), float(ci.high)] if ci else None,
        "retok_mismatch_ids": [r["i"] for r in recs if not r["retok_match"]],
        "kill_rule_fires": (k_mis / n > KILL_RULE_MISMATCH_RATE) if (n and beta == 32) else None,
        "payload_tokens": {"mean": float(pay_tok.mean()), "median": float(np.median(pay_tok)),
                           "min": int(pay_tok.min()), "max": int(pay_tok.max())},
        "raw_bits_per_payload_token": raw_bits / pay_steps,
        "effective_bits_per_payload_token": 16 * n / pay_steps,
        "payload_steps": pay_steps,
        "vanished_per_payload_step": sum(r["n_vanished_payload"] for r in recs) / pay_steps,
        "messages_with_payload_vanishing": sum(r["n_vanished_payload"] > 0 for r in recs),
        "cover_steps": cover_steps,
        "vanished_per_cover_step": (sum(r["n_vanished_cover"] for r in recs) / cover_steps) if cover_steps else None,
        "eos_masked_step_fraction": sum(r["eos_masked_steps"] for r in recs) / pay_steps,
        "messages_with_eos_masking": sum(r["eos_masked_steps"] > 0 for r in recs),
        "eos_mass_removed_mean_per_message": float(np.mean([r["eos_mass_removed"] for r in recs])),
        "stop_reasons": {s: sum(r["stop_reason"] == s for r in recs) for s in ("eos", "max_new_tokens")},
        "cover_tokens_mean": float(np.mean([r["n_cover_tokens"] for r in recs])),
        "encode_s_mean": float(np.mean([r["encode_s"] for r in recs])),
        "decode_text_s_mean": float(np.mean([r["decode_text_s"] for r in recs])),
    }


def make_figures(runs: dict, figdir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, ink2, grid, blue, surface = "#0b0b0b", "#52514e", "#e6e5e1", "#2a78d6", "#fcfcfb"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": ink2, "axes.labelcolor": ink,
                         "xtick.color": ink2, "ytick.color": ink2, "axes.titlecolor": ink,
                         "figure.facecolor": surface, "axes.facecolor": surface})
    figdir.mkdir(parents=True, exist_ok=True)
    betas = sorted(runs, key=int)

    # 1) tokens needed for the 16-bit payload, one small multiple per beta
    fig, axes = plt.subplots(1, len(betas), figsize=(3 * len(betas), 2.8), sharey=True)
    hi = max(max(r["n_payload_tokens"] for r in runs[b]["records"]) for b in betas)
    bins = np.arange(0.5, hi + 1.5)
    for ax, b in zip(np.atleast_1d(axes), betas):
        x = [r["n_payload_tokens"] for r in runs[b]["records"]]
        ax.hist(x, bins=bins, color=blue, edgecolor=surface, linewidth=2)
        ax.set_title(f"β = {b}  (median {np.median(x):g})", fontsize=10)
        ax.set_xlabel("payload tokens")
        ax.grid(axis="y", color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    np.atleast_1d(axes)[0].set_ylabel("messages")
    n_msgs = len(runs[betas[0]]["records"])
    fig.suptitle(f"Tokens needed to send the 16-bit payload ({n_msgs} messages per β)", color=ink, fontsize=11)
    fig.tight_layout()
    fig.savefig(figdir / "01_tokens_for_payload.png", dpi=150)
    plt.close(fig)

    # 2) vanishing rate and capacity by beta: two panels, one measure each (no dual axis)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.2))
    labels = [f"β={b}" for b in betas]
    van = [runs[b]["summary"]["vanished_per_payload_step"] for b in betas]
    cap = [runs[b]["summary"]["raw_bits_per_payload_token"] for b in betas]
    for ax, vals, title, fmt in ((a1, van, "Vanished tokens per payload step", "{:.3g}"),
                                 (a2, cap, "Bits carried per payload token", "{:.2f}")):
        bars = ax.bar(labels, vals, color=blue, width=0.6)
        for bar, v in zip(bars, vals):
            ax.annotate(fmt.format(v), (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha="center", va="bottom", fontsize=9, color=ink, xytext=(0, 2),
                        textcoords="offset points")
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(figdir / "01_vanishing_and_capacity.png", dpi=150)
    plt.close(fig)


def main(limit: int | None, out: Path, figdir: Path) -> None:
    lm = LocalLM()
    keys_s, keys_r = pair_from_seed(SEED)
    n = limit or len(SUBJECTS) * MESSAGES_PER_PROMPT
    msgs = make_messages(n, lm.cfg.message_bits)
    runs: dict[str, dict] = {}
    t_start = time.perf_counter()
    for run_index, (beta, max_new) in enumerate(RUNS):
        cfg = MeteorConfig(beta=beta, max_new_tokens=max_new)
        recs = []
        for i in range(n):
            recs.append(run_one(lm, keys_s, keys_r, cfg, run_index, i, msgs[i]))
            if (i + 1) % 20 == 0 or i + 1 == n:
                print(f"beta={beta}: {i + 1}/{n}  elapsed {time.perf_counter() - t_start:.0f}s", flush=True)
        runs[str(beta)] = {"summary": summarise(recs, beta, max_new), "records": recs}
        print(f"beta={beta} summary: {runs[str(beta)]['summary']}", flush=True)
        write_result_json(out, {"runs": runs, "complete": False}, lm.cfg, **params(n))  # checkpoint
    write_result_json(out, {"runs": runs, "complete": True,
                            "total_wall_s": round(time.perf_counter() - t_start, 1)}, lm.cfg, **params(n))
    make_figures(runs, figdir)
    print(f"wrote {out} and figures in {figdir}")


def params(n: int) -> dict:
    return {"experiment": "01_meteor_roundtrip", "n_messages": n, "runs_beta_maxnew": RUNS,
            "msg_id_stride": MSG_ID_STRIDE, "subjects": SUBJECTS,
            "messages_per_prompt": MESSAGES_PER_PROMPT, "key_seed": SEED, "message_seed": SEED,
            "kill_rule_mismatch_rate": KILL_RULE_MISMATCH_RATE}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="messages per beta (smoke test)")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "01_meteor_roundtrip.json")
    ap.add_argument("--figdir", type=Path, default=REPO_ROOT / "results" / "figs")
    a = ap.parse_args()
    main(a.limit, a.out, a.figdir)
