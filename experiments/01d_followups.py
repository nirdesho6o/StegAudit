"""Experiment 01d (EXPLORATORY, NOT PRE-REGISTERED): two follow-ups on experiment 01.

Requested by Nirdesh on 2026-11-15, after 01's results. No PREREG change.

1. Cover-text vanishing at beta = 32. Experiment 01 found 84 vanished tokens in
   the cover text of 9/200 messages. Here those 9 messages are replayed (same
   KV-cache path, EOS NOT masked, as in the cover phase). At every cover step
   where quantisation at beta = 32 gives at least one zero-width token, we log
   the top-1 probability and the entropy of the top-k distribution. Tested
   claim: "they occur at near-deterministic steps (top-1 p > 0.999)".
   Check: the replayed vanished count must equal 01's recorded count (84).

2. Payload-only retokenisation mismatch rate. A mismatch counts as
   "payload-relevant" if the first token where retokenisation differs from
   the Sender's ids lies inside the payload (index < n_payload_tokens).
   Descriptive only; the registered kill rule (on all mismatches) is unchanged.

Output: results/01d_followups.json
"""

from __future__ import annotations

import json
import time

import numpy as np
from scipy.stats import binomtest

from stegaudit.lm import LocalLM
from stegaudit.provenance import REPO_ROOT, write_result_json
from stegaudit.quantize import quantize

SRC = REPO_ROOT / "results" / "01_meteor_roundtrip.json"
OUT = REPO_ROOT / "results" / "01d_followups.json"
NEAR_DET = 0.999


def entropy_bits(p):
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def cover_vanishing_steps(lm, rec, beta):
    """Replay one message; return per-step info for cover steps with vanished tokens."""
    sess = lm.start(lm.chat_prompt_ids(rec["prompt"]))
    n_pay, ids = rec["n_payload_tokens"], rec["ids"]
    for t in ids[:n_pay]:
        sess.advance(t)
    steps = []
    cover_ids = ids[n_pay:] + ([None] if rec["stop_reason"] == "eos" else [])  # final step that chose EOS
    for j, t in enumerate(cover_ids):
        d = sess.dist(mask_eos=False)
        qd = quantize(d.probs, beta)
        if qd.n_vanished:
            steps.append({"cover_step": j, "token_index": n_pay + j, "n_vanished": qd.n_vanished,
                          "top1_p": float(d.probs.max()), "entropy_bits": entropy_bits(d.probs),
                          "min_p": float(d.probs.min())})
        if t is not None:
            sess.advance(t)
    return steps


def first_diff(a, b):
    return next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), min(len(a), len(b)))


def main():
    data = json.loads(SRC.read_text())
    runs = data["results"]["runs"]
    lm = LocalLM()
    assert lm.cfg.model_revision == data["meta"]["model_revision"]
    t0 = time.perf_counter()

    # 1. cover vanishing at beta = 32
    recs = [r for r in runs["32"]["records"] if r["n_vanished_cover"] > 0]
    steps = []
    for r in recs:
        for s in cover_vanishing_steps(lm, r, 32):
            steps.append({"i": r["i"], **s})
    replayed = sum(s["n_vanished"] for s in steps)
    recorded = sum(r["n_vanished_cover"] for r in recs)
    top1 = np.array([s["top1_p"] for s in steps])
    ent = np.array([s["entropy_bits"] for s in steps])
    part1 = {
        "messages": len(recs), "steps_with_vanishing": len(steps),
        "vanished_tokens_replayed": replayed, "vanished_tokens_recorded_in_01": recorded,
        "counts_match": replayed == recorded,
        "top1_p": {"min": float(top1.min()), "median": float(np.median(top1)), "max": float(top1.max())},
        "entropy_bits": {"min": float(ent.min()), "median": float(np.median(ent)), "max": float(ent.max())},
        "steps_with_top1_above_0.999": int((top1 > NEAR_DET).sum()),
        "claim_near_deterministic_all_steps": bool((top1 > NEAR_DET).all()),
        "steps": steps,
    }
    print({k: v for k, v in part1.items() if k != "steps"}, flush=True)

    # 2. payload-only retokenisation mismatches (tokenizer only, no model replay)
    part2 = {}
    for beta, run in runs.items():
        n, k_all, k_pay, which = 0, 0, 0, []
        for r in run["records"]:
            n += 1
            re_ids = lm.text_to_ids(r["text"])
            if re_ids == r["ids"]:
                continue
            k_all += 1
            j = first_diff(re_ids, r["ids"])
            if j < r["n_payload_tokens"]:
                k_pay += 1
                which.append(r["i"])
        ci = binomtest(k_pay, n).proportion_ci(confidence_level=0.95, method="exact")
        part2[beta] = {"n": n, "mismatches_all": k_all, "mismatches_in_payload": k_pay,
                       "payload_mismatch_rate": k_pay / n, "ci95_exact": [float(ci.low), float(ci.high)],
                       "payload_mismatch_ids": which}
    tot_n = sum(v["n"] for v in part2.values())
    tot_k = sum(v["mismatches_in_payload"] for v in part2.values())
    ci = binomtest(tot_k, tot_n).proportion_ci(confidence_level=0.95, method="exact")
    part2["all_betas"] = {"n": tot_n, "mismatches_in_payload": tot_k, "rate": tot_k / tot_n,
                          "ci95_exact": [float(ci.low), float(ci.high)]}
    print(part2, flush=True)

    write_result_json(OUT, {"exploratory": True, "preregistered": False,
                            "cover_vanishing_beta32": part1, "payload_retok_mismatch": part2,
                            "source_git_commit": data["meta"]["git_commit"],
                            "wall_s": round(time.perf_counter() - t0, 1)},
                      lm.cfg, experiment="01d_followups", near_deterministic_threshold=NEAR_DET)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
