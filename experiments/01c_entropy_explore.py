"""Experiment 01c (EXPLORATORY, NOT PRE-REGISTERED): per-step entropy and efficiency.

Added after experiment 01 had started, at Nirdesh's request (2026-11-15).
It changes no pre-registered prediction; see PREREG.md -> Deviations.

What it measures
----------------
For each payload step of the first N_PER_BETA messages of every beta run in
experiment 01:
  * H = Shannon entropy (bits) of the distribution Meteor sampled from at that
    step: the top-k (k = 50) distribution, renormalised, EOS masked, float64.
    H is of this top-k distribution, not of the full-vocabulary model.
  * n = message bits that step carried (common-prefix length, from 01's JSON).
  * efficiency_raw    = sum(n) / sum(H) over all payload steps,
    efficiency_capped = same with n capped so a message never counts more than
                        its L = 16 payload bits (the last step can overshoot).
  * per-step ratio n / H (steps with H > 0), and the correlation of n with H.
"Efficiency" here is this project's own ratio. It is NOT claimed to equal
any efficiency defined in the Meteor paper (not checked against the PDF).

How
---
No new encoding. The model is replayed over the token ids saved by experiment
01 (prefill the prompt, then one token at a time, EOS masked), which gives
bit-identical distributions to the encoder's (tested in Phase 1). As a sanity
check, every saved token must lie in that step's top-k.

Output: results/01c_entropy_explore.json, results/figs/01c_bits_vs_entropy.png
Usage:  python experiments/01c_entropy_explore.py [--n 30]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from stegaudit.lm import LocalLM
from stegaudit.provenance import REPO_ROOT, write_result_json

N_PER_BETA = 30
L = 16


def entropy_bits(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def replay_entropies(lm: LocalLM, prompt: str, payload_ids: list[int]) -> list[float]:
    sess = lm.start(lm.chat_prompt_ids(prompt))
    hs = []
    for tok in payload_ids:
        d = sess.dist(mask_eos=True)
        if tok not in set(d.ids.tolist()):
            raise AssertionError("saved token not in replayed top-k: replay does not match encoder")
        hs.append(entropy_bits(d.probs))
        sess.advance(tok)
    return hs


def capped(bits_per_step: list[int]) -> list[int]:
    out, total = [], 0
    for n in bits_per_step:
        c = max(0, min(n, L - total))
        out.append(c)
        total += c
    return out


def make_figure(steps32: list[tuple[float, int]], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, ink2, grid, blue, orange, surface = "#0b0b0b", "#52514e", "#e6e5e1", "#2a78d6", "#eb6834", "#fcfcfb"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": ink2, "axes.labelcolor": ink,
                         "xtick.color": ink2, "ytick.color": ink2, "figure.facecolor": surface,
                         "axes.facecolor": surface})
    h = np.array([s[0] for s in steps32])
    n = np.array([s[1] for s in steps32], dtype=float)
    jitter = np.random.default_rng(0).uniform(-0.15, 0.15, len(n))
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.scatter(h, n + jitter, s=10, color=blue, alpha=0.45, linewidths=0, label="one payload step")
    edges = np.linspace(0, h.max() + 1e-9, 9)
    idx = np.digitize(h, edges) - 1
    centres = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1) if np.any(idx == i)]
    means = [n[idx == i].mean() for i in range(len(edges) - 1) if np.any(idx == i)]
    ax.plot(centres, means, color=orange, linewidth=2, marker="o", markersize=5, label="mean bits per entropy bin")
    ax.set_xlabel("entropy of the top-k distribution at that step (bits)")
    ax.set_ylabel("message bits carried by the step")
    ax.set_title("β = 32: bits carried vs entropy (exploratory, not pre-registered)", fontsize=10, color=ink)
    ax.grid(color=grid, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(n_per_beta: int, inp: Path, out: Path, fig: Path) -> None:
    data = json.loads(inp.read_text())
    lm = LocalLM()
    assert lm.cfg.model_revision == data["meta"]["model_revision"]
    t0 = time.perf_counter()
    per_beta, steps32 = {}, []
    for beta, run in data["results"]["runs"].items():
        H, N, NC = [], [], []
        for r in run["records"][:n_per_beta]:
            hs = replay_entropies(lm, r["prompt"], r["ids"][:r["n_payload_tokens"]])
            H += hs
            N += r["bits_per_token"]
            NC += capped(r["bits_per_token"])
            if beta == "32":
                steps32 += list(zip(hs, r["bits_per_token"]))
        H, N, NC = np.array(H), np.array(N, dtype=float), np.array(NC, dtype=float)
        pos = H > 0
        ratio = N[pos] / H[pos]
        per_beta[beta] = {
            "n_messages": min(n_per_beta, len(run["records"])), "payload_steps": int(len(H)),
            "entropy_mean_bits": float(H.mean()), "entropy_median_bits": float(np.median(H)),
            "bits_mean": float(N.mean()),
            "efficiency_raw": float(N.sum() / H.sum()),
            "efficiency_capped": float(NC.sum() / H.sum()),
            "ratio_n_over_H": {"mean": float(ratio.mean()), "median": float(np.median(ratio)),
                               "p10": float(np.percentile(ratio, 10)), "p90": float(np.percentile(ratio, 90))},
            "steps_with_H_zero": int((~pos).sum()),
            "pearson_n_H": float(np.corrcoef(N, H)[0, 1]),
        }
        print(f"beta={beta}: {per_beta[beta]}", flush=True)
    make_figure(steps32, fig)
    write_result_json(out, {"exploratory": True, "preregistered": False,
                            "source": str(inp.resolve().relative_to(REPO_ROOT)),
                            "source_git_commit": data["meta"]["git_commit"],
                            "per_beta": per_beta, "wall_s": round(time.perf_counter() - t0, 1)},
                      lm.cfg, experiment="01c_entropy_explore", n_per_beta=n_per_beta, L=L)
    print(f"wrote {out} and {fig}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N_PER_BETA)
    ap.add_argument("--in", dest="inp", type=Path, default=REPO_ROOT / "results" / "01_meteor_roundtrip.json")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "01c_entropy_explore.json")
    ap.add_argument("--fig", type=Path, default=REPO_ROOT / "results" / "figs" / "01c_bits_vs_entropy.png")
    a = ap.parse_args()
    main(a.n, a.inp, a.out, a.fig)
