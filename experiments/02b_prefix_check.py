"""Experiment 02b (EXPLORATORY, NOT PRE-REGISTERED): is the prefix log-prob signal a boundary artefact?

In experiment 02 the mean log-prob of the payload prefix gave AUROC 0.447
[0.391, 0.505] (KS p = 0.022): Meteor prefixes look slightly less likely.
Hypothesis (not assumed): the prefix ends exactly where the last payload bits
are sent, and steps that send many bits tend to be low-probability tokens.
So the LAST prefix token is selected to be unlikely, and that pulls the mean
down. A monitor would not know where the payload ends; the experiment hands
it this boundary.

Checks (same 200 + 200 texts, same pairing as 02):
  1. prefix mean log-prob WITHOUT the last prefix token (both groups, n - 1 tokens);
  2. log-prob of the last prefix token alone (Meteor vs normal at the same position);
  3. a fixed window: first 10 tokens of every text, independent of the payload boundary.
If the artefact explains the signal: 1 and 3 give AUROC ~ 0.5, and 2 is clearly below 0.5.

Output: results/02b_prefix_check.json
"""

from __future__ import annotations

import json
import time

import numpy as np

from stegaudit.detect import auroc_ci, ks, token_scores
from stegaudit.lm import LocalLM
from stegaudit.provenance import REPO_ROOT, write_result_json

R = REPO_ROOT / "results"
FIXED = 10


def main():
    met = json.loads((R / "01_meteor_roundtrip.json").read_text())["results"]["runs"]["32"]["records"]
    nor = json.loads((R / "02_texts_normal.json").read_text())["results"]["texts"]
    lm = LocalLM()
    t0 = time.perf_counter()
    out = {k: ([], []) for k in ("prefix_all", "prefix_without_last", "last_token", "first10")}
    skipped = 0
    for m, n in zip(met, nor):
        k = m["n_payload_tokens"]
        L = max(k, FIXED)
        if len(n["ids"]) < L or len(m["ids"]) < L:
            skipped += 1
            continue
        sm = token_scores(lm, m["prompt"], m["ids"][:L])["logp_full"]
        sn = token_scores(lm, n["prompt"], n["ids"][:L])["logp_full"]
        for g, s in ((0, sm), (1, sn)):
            out["prefix_all"][g].append(float(s[:k].mean()))
            out["prefix_without_last"][g].append(float(s[:k - 1].mean()) if k > 1 else float("nan"))
            out["last_token"][g].append(float(s[k - 1]))
            out["first10"][g].append(float(s[:FIXED].mean()))
    table = {}
    for name, (pos, neg) in out.items():
        pos, neg = np.array(pos), np.array(neg)
        keep = ~(np.isnan(pos) | np.isnan(neg))
        table[name] = {**auroc_ci(pos[keep], neg[keep]), **ks(pos[keep], neg[keep]),
                       "mean_meteor": float(pos[keep].mean()), "mean_normal": float(neg[keep].mean())}
        print(f"{name:<22} AUROC {table[name]['auroc']:.3f} CI {np.round(table[name]['ci95'], 3)} "
              f"KS p {table[name]['p_value']:.3g}  mean M {table[name]['mean_meteor']:.3f} "
              f"N {table[name]['mean_normal']:.3f}", flush=True)
    write_result_json(R / "02b_prefix_check.json",
                      {"exploratory": True, "preregistered": False, "table": table,
                       "pairs_used": len(out["prefix_all"][0]), "pairs_skipped_too_short": skipped,
                       "wall_s": round(time.perf_counter() - t0, 1)},
                      lm.cfg, experiment="02b_prefix_check", fixed_window=FIXED)
    print(f"pairs used {len(out['prefix_all'][0])}, skipped {skipped}")


if __name__ == "__main__":
    main()
