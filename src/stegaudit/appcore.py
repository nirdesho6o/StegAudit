"""Helpers for the Streamlit app (app/streamlit_app.py). No algorithm lives in the app file.

Idea in plain words
-------------------
Tab 4 (audit game) lets a viewer change budget, stake inequality (Gini), audit
detection power d, false-flag rate f and penalty P, and see:
  * how each auditor policy spreads its coverage over the stakes,
  * how much stake each colluder gets through undetected under each policy,
  * how many decisions a best-responding colluder attacks (ties go to the
    auditor, PREREG ed19351), and
  * how many audits a test needs to catch a uniform-eps colluder at (d, f)
    (the P3a definition, PREREG 7088dd5).
Everything is computed by simulate.py / audit_lp.py, the same code experiment
04 uses, so the app cannot disagree with the experiment.

Security assumption and what breaks
-----------------------------------
Same as simulate.py: colluders know the coverage and stakes; d, f, P are
fixed numbers; "suspicion" auditing = random auditing because the channel is
undetectable from transcripts. If a transcript detector worked, the
suspicion policy shown here would understate what an auditor can do.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from stegaudit.provenance import REPO_ROOT
from stegaudit.simulate import (attack_best_response, attack_top_k, attack_uniform, audits_to_detect,
                                count_ties, coverage, expected_undetected, stakes)

POLICIES = ["uniform", "suspicion", "stake_proportional", "m1", "m2"]
POLICY_LABELS = {"uniform": "Uniform", "suspicion": "Suspicion (= random)",
                 "stake_proportional": "Stake-proportional", "m1": "LP M1", "m2": "LP M2"}
EPS_VALUES = [0.01, 0.05]          # registered colluder rates for 04 (PREREG 7a7fa45)
P_MULTS = [0, 1, 10, 100]          # P = mult x mean(v), as in 04
T_DEFAULT = 1000


def colluder_labels(eps_values=EPS_VALUES) -> list[str]:
    return (["best response"] + [f"top-k (k = {e:g}·T)" for e in eps_values]
            + [f"uniform ε = {e:g}" for e in eps_values])


def audit_game(dist: str, gini_target: float, b_frac: float, d: float, p_mult: float,
               T: int = T_DEFAULT, seed: int = 0, eps_values=EPS_VALUES) -> dict:
    """One instance of the audit game: every policy against every colluder (closed form)."""
    v = stakes(dist, gini_target, T, np.random.default_rng(seed))
    P, B = p_mult * float(v.mean()), b_frac * T
    out = {"v": v, "P": P, "B": B, "total_stake": float(v.sum()), "policies": {}}
    for pol in POLICIES:
        c = coverage(pol, v, B, d, P)
        colluders = {"best response": attack_best_response(c, v, d, P)}
        for e in eps_values:
            colluders[f"top-k (k = {e:g}·T)"] = attack_top_k(v, int(round(e * T)))
        for e in eps_values:
            colluders[f"uniform ε = {e:g}"] = attack_uniform(T, e)
        out["policies"][pol] = {
            "c": c,
            "budget_used": float(c.sum()),
            "attacked": int(colluders["best response"].sum()),
            "ties": count_ties(c, v, d, P),
            "undetected": {k: expected_undetected(a, c, v, d) for k, a in colluders.items()},
        }
    return out


def deterrence_boundary(v, d: float, P: float) -> np.ndarray:
    """Coverage at which attacking decision i stops paying: c_i = v_i / (d (v_i + P)), capped at 1."""
    v = np.asarray(v, float)
    return np.minimum(1.0, v / (d * (v + P)))


def audits_needed(d: float, f: float, eps_values=EPS_VALUES) -> dict:
    """Audits to catch a uniform-eps colluder (P3a definition; level 0.05, power 0.8 when f > 0)."""
    return {e: float(audits_to_detect(e, d, fpr=f)) for e in eps_values}


def load_d_estimate(path: Path = REPO_ROOT / "results" / "d_estimate.json") -> list[dict]:
    """Phase 2's (d, f) pairs, one per Sentinel; [] if Phase 2 has not produced them yet."""
    if not Path(path).exists():
        return []
    return json.loads(Path(path).read_text())["results"]["pairs"]


# ---------------- tab 1: Meteor live ----------------

DEMO_SECRET = "C"
DEMO_PROMPT = "Write a short study tip about chemistry."
DEMO_BETAS = [8, 12, 16, 32]
DEMO_KEY_SEED = 2026                 # demo keys only (Party.from_seed: never for real use)
DEMO_MSG_ID = 0
DEMO_TAG_BITS = 8
DEMO_MAX_NEW_TOKENS = 60
MAX_SECRET_CHARS = 4


def secret_to_bits(secret: str) -> list[int]:
    """UTF-8 bytes of the secret, 8 bits each, most significant bit first."""
    return [(byte >> (7 - k)) & 1 for byte in secret.encode("utf-8") for k in range(8)]


def bits_to_secret(bits) -> str:
    data = bytes(int("".join(map(str, bits[i:i + 8])), 2) for i in range(0, len(bits) - len(bits) % 8, 8))
    return data.decode("utf-8", errors="replace")


def key_exchange_view(seed: int = DEMO_KEY_SEED) -> dict:
    """What each party sends (public keys) and what each ends up with (key fingerprints)."""
    from stegaudit.keys import Party
    sender, receiver, eve = (Party.from_seed(seed, n) for n in ("sender", "receiver", "eve"))
    ks, kr = sender.derive_keys(receiver.public_bytes()), receiver.derive_keys(sender.public_bytes())
    ke = eve.derive_keys(sender.public_bytes())          # Eve pairs her own key with the Sender's public key
    return {"sender_public": sender.public_bytes().hex(), "receiver_public": receiver.public_bytes().hex(),
            "sender_fingerprint": ks.fingerprint(), "receiver_fingerprint": kr.fingerprint(),
            "eve_fingerprint": ke.fingerprint(), "keys": (ks, kr, ke)}


def meteor_demo(lm, secret: str, beta: int, prompt: str = DEMO_PROMPT, seed: int = DEMO_KEY_SEED,
                max_new_tokens: int = DEMO_MAX_NEW_TOKENS) -> dict:
    """Encode `secret` into a study tip, then decode with the right key and with Eve's key.

    Returns only JSON-friendly values (lists, ints, strings) so the same dict can be saved as the
    demo-mode example (experiments/app_demo_data.py) and rendered from disk without the model.
    """
    from stegaudit.config import MeteorConfig
    from stegaudit.meteor import decode, encode
    if not 1 <= len(secret) <= MAX_SECRET_CHARS:
        raise ValueError(f"secret must have 1-{MAX_SECRET_CHARS} characters")
    msg = secret_to_bits(secret)
    cfg = MeteorConfig(beta=beta, payload_bits=len(msg) + DEMO_TAG_BITS, tag_bits=DEMO_TAG_BITS,
                       max_new_tokens=max_new_tokens)
    kx = key_exchange_view(seed)
    ks, kr, ke = kx.pop("keys")
    trace: list = []
    enc = encode(lm, ks, prompt, msg, DEMO_MSG_ID, cfg, trace=trace)
    right, wrong = (decode(lm, k, prompt, enc.text, DEMO_MSG_ID, cfg) for k in (kr, ke))
    tok = lambda i: lm.ids_to_text([int(i)])            # noqa: E731
    steps = [{"step": t["step"], "chunk": t["chunk"], "mask": t["mask"], "x": t["x"], "pos": t["pos"],
              "token": t["token"], "n_bits": t["n_bits"], "ptr_before": t["ptr_before"],
              "ids": [int(i) for i in t["ids"]], "tokens": [tok(i) for i in t["ids"]],
              "probs": [float(p) for p in t["probs"]], "lo": [int(x) for x in t["lo"]],
              "widths": [int(w) for w in t["widths"]]} for t in trace]
    return {"secret": secret, "message_bits": msg, "payload": enc.payload, "beta": beta, "prompt": prompt,
            "payload_bits": cfg.payload_bits, "tag_bits": cfg.tag_bits, "top_k": cfg.top_k,
            "key_exchange": kx, "token_texts": [tok(i) for i in enc.ids], "ids": [int(i) for i in enc.ids],
            "text": enc.text, "n_payload_tokens": enc.n_payload_tokens, "steps": steps,
            "decode_right": {"ok": right.ok, "failure": right.failure,
                             "secret": bits_to_secret(right.message_bits) if right.ok else None},
            "decode_wrong": {"ok": wrong.ok, "failure": wrong.failure,
                             "payload_bits": [int(b) for b in wrong.payload_bits]}}


def step_detail(step: dict, beta: int) -> dict:
    """Binary view of one payload step: chosen interval, x, common prefix, bits sent."""
    pos = step["pos"]
    lo, hi = step["lo"][pos], step["lo"][pos] + step["widths"][pos]
    b = lambda v: format(v, f"0{beta}b")                 # noqa: E731
    n = step["n_bits"]
    return {"chunk": b(step["chunk"]), "mask": b(step["mask"]), "x": b(step["x"]),
            "lo": b(lo), "hi_minus_1": b(hi - 1), "n_bits": n, "bits_sent": b(step["chunk"])[:n],
            "prob": step["probs"][pos], "width": hi - lo}


# ---------------- tab 2: detectability (saved results of experiments 01, 02, 02b) ----------------

DETECTOR_LABELS = {"mean_logprob": "mean token log-prob", "vanishing_count": "vanishing tokens (p < 2^-β)",
                   "eos_sum": "EOS probability mass", "outside_topk_count": "tokens outside top-k"}


def _results(name: str, results_dir: Path) -> dict:
    return json.loads((Path(results_dir) / name).read_text())["results"]


def detector_rows(results_dir: Path = REPO_ROOT / "results") -> list[dict]:
    """Every experiment-02 detector at beta = 32: AUROC (Meteor = positive) with its 95 % CI."""
    local = _results("02_local.json", results_dir)["table"]
    rows = []
    for view in ("whole", "prefix"):
        for det, label in DETECTOR_LABELS.items():
            t = local[f"beta32/{view}/{det}"]
            counts = det in ("vanishing_count", "outside_topk_count")    # "events" only means something here
            rows.append({"detector": label, "view": view, "auroc": t["auroc"], "ci95": t["ci95"],
                         "events_meteor": t.get("nonzero_meteor") if counts else None,
                         "events_normal": t.get("nonzero_normal") if counts else None})
    jpath = Path(results_dir) / "02_judge.json"
    if jpath.exists():
        for key, t in _results("02_judge.json", results_dir)["table"].items():
            model, view = key.split("/")
            rows.append({"detector": f"LLM judge ({model})", "view": view, "auroc": t["auroc"], "ci95": t["ci95"],
                         "events_meteor": None, "events_normal": None})
    return rows


def text_pair(i: int, results_dir: Path = REPO_ROOT / "results") -> dict:
    """Meteor text i (beta = 32, experiment 01) and normal text i (02, same prompt), with per-text scores."""
    met = _results("01_meteor_roundtrip.json", results_dir)["runs"]["32"]["records"][i]
    nor = _results("02_texts_normal.json", results_dir)["texts"][i]
    if met["prompt"] != nor["prompt"]:
        raise AssertionError("experiments 01 and 02 are paired by index; prompts must match")
    per = _results("02_local.json", results_dir)["per_text"]["beta32/whole/mean_logprob"]
    judge = {"meteor": None, "normal": None}
    jpath = Path(results_dir) / "02_judge.json"
    if jpath.exists():
        for r in _results("02_judge.json", results_dir)["rows"]:
            if r["i"] == met["i"] and r["condition"] in ("meteor/whole", "normal/whole"):
                judge[r["condition"].split("/")[0]] = r["score"]
    return {"prompt": met["prompt"], "meteor_text": met["text"], "normal_text": nor["text"],
            "n_payload_tokens": met["n_payload_tokens"], "meteor_logprob": per["meteor"][i],
            "normal_logprob": per["normal"][i], "judge": judge, "n_texts": 200,
            "meteor_on_left": bool(np.random.default_rng(i).integers(2))}   # fixed per i: a fair quiz
