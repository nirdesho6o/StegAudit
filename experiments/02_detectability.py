"""Experiment 02: can a monitor tell Meteor texts from normal texts? (design: PREREG.md)

Stages (run in this order):
  generate --group normal|8|12|16   local texts -> results/02_texts_<group>.json
  score                             local detectors (a)-(c), whole text + prefix, every beta
                                    -> results/02_local.json
  judge-pilot                       20 Adaption calls (5 whole + 5 prefix beta-32 Meteor texts x 2
                                    judges) -> results/02_judge_pilot.json + cost estimate
  judge [--max-credits 0.5]         full LLM-judge set (800 texts x 2 judges), only if the
                                    estimate is below --max-credits -> results/02_judge.json
  report                            summary + figures -> results/02_detectability.json, figs

Meteor texts at beta = 32 come from experiment 01 (results/01_meteor_roundtrip.json).
Remote calls are cached in cache/ (rerunning costs nothing) and logged in results/credits.json.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from stegaudit.config import MeteorConfig
from stegaudit.detect import (JUDGE_INSTRUCTIONS, auroc_ci, eos_sum, judge_input, ks, mean_logprob,
                              outside_topk_count, parse_judge, sample_plain, token_scores, vanishing_count)
from stegaudit.keys import pair_from_seed
from stegaudit.provenance import REPO_ROOT, write_result_json
from stegaudit.remote import AdaptionClient, cached_token_stats, estimate_cost_from_cache

# Output folder; STEGAUDIT_02_DIR overrides it for smoke tests (input from 01 always from results/).
R = Path(os.environ.get("STEGAUDIT_02_DIR", REPO_ROOT / "results"))
SRC01 = REPO_ROOT / "results" / "01_meteor_roundtrip.json"
SWEEP_RUN_INDEX = {8: 4, 12: 5, 16: 6}          # msg_id = 10 000 * run_index + i (PREREG)
NORMAL_SEED = 1
MAX_NEW_TOKENS = 120
# Detector beta sweep (8, 12, 16) CUT on 2026-11-15 for the deadline (PREREG Deviations).
# The generate stage still accepts --group 8|12|16 for future work, but score/report use BETAS only.
BETAS = [32]
JUDGES = ["qwen3.5-397b", "glm-5.2"]
JUDGE_MAX_OUTPUT_TOKENS = 4096
JUDGE_EXPERIMENT = "02_judge"
PARSE_FAILURE_STOP = 0.05
N_BOOT = 2000


def load01():
    d = json.loads(SRC01.read_text())
    return d, d["results"]["runs"]


# ---------------- generate ----------------

def stage_generate(group: str, limit: int | None) -> None:
    from stegaudit.lm import LocalLM
    from stegaudit.meteor import encode
    _, runs = load01()
    base = runs["32"]["records"][:limit] if limit else runs["32"]["records"]
    lm = LocalLM()
    out, t0 = [], time.perf_counter()
    if group == "normal":
        rng = np.random.default_rng(NORMAL_SEED)
        for r in base:
            ids = sample_plain(lm, r["prompt"], rng, MAX_NEW_TOKENS)
            out.append({"i": r["i"], "prompt": r["prompt"], "ids": ids, "text": lm.ids_to_text(ids)})
            if len(out) % 20 == 0:
                print(f"normal {len(out)}/{len(base)} {time.perf_counter() - t0:.0f}s", flush=True)
    else:
        beta = int(group)
        keys_s, _ = pair_from_seed(json.loads(SRC01.read_text())["meta"]["params"]["key_seed"])
        cfg = MeteorConfig(beta=beta, max_new_tokens=MAX_NEW_TOKENS)
        for r in base:
            msg_id = 10_000 * SWEEP_RUN_INDEX[beta] + r["i"]
            enc = encode(lm, keys_s, r["prompt"], r["message"], msg_id, cfg)
            out.append({"i": r["i"], "prompt": r["prompt"], "msg_id": msg_id, "message": r["message"],
                        "ids": enc.ids, "text": enc.text, "n_payload_tokens": enc.n_payload_tokens,
                        "eos_masked_steps": enc.eos_masked_steps, "stop_reason": enc.stop_reason})
            if len(out) % 20 == 0:
                print(f"beta={beta} {len(out)}/{len(base)} {time.perf_counter() - t0:.0f}s", flush=True)
    path = R / f"02_texts_{group}.json"
    write_result_json(path, {"group": group, "texts": out, "wall_s": round(time.perf_counter() - t0, 1)},
                      lm.cfg, experiment="02_generate", group=group, normal_seed=NORMAL_SEED,
                      max_new_tokens=MAX_NEW_TOKENS, limit=limit)
    print(f"wrote {path}")


def load_texts():
    """{'normal': [...], 32: [...], 8: [...], ...} aligned by index i."""
    _, runs = load01()
    texts = {32: [{k: r[k] for k in ("i", "prompt", "ids", "text", "n_payload_tokens")}
                  for r in runs["32"]["records"]]}
    for g in ["normal"] + [b for b in BETAS if b != 32]:
        p = R / f"02_texts_{g}.json"
        texts[g] = json.loads(p.read_text())["results"]["texts"]
    n = min(len(v) for v in texts.values())
    return {k: v[:n] for k, v in texts.items()}


# ---------------- score (local detectors) ----------------

def stage_score() -> None:
    from stegaudit.lm import LocalLM
    lm = LocalLM()
    texts = load_texts()
    t0 = time.perf_counter()
    scores = {g: [token_scores(lm, t["prompt"], t["ids"]) for t in texts[g]] for g in texts}
    print(f"scored {sum(len(v) for v in scores.values())} texts in {time.perf_counter() - t0:.0f}s", flush=True)

    table, per_text = {}, {}
    for beta in BETAS:
        met, nor = scores[beta], scores["normal"]
        npay = [t["n_payload_tokens"] for t in texts[beta]]
        for view in ("whole", "prefix"):
            lim = (lambda i: None) if view == "whole" else (lambda i: npay[i])
            stats = {
                "mean_logprob": ([mean_logprob(s, lim(i)) for i, s in enumerate(met)],
                                 [mean_logprob(s, lim(i)) for i, s in enumerate(nor)]),
                "vanishing_count": ([vanishing_count(s, beta, lim(i)) for i, s in enumerate(met)],
                                    [vanishing_count(s, beta, lim(i)) for i, s in enumerate(nor)]),
                "eos_sum": ([eos_sum(s, lim(i)) for i, s in enumerate(met)],
                            [eos_sum(s, lim(i)) for i, s in enumerate(nor)]),
                "outside_topk_count": ([outside_topk_count(s, lim(i)) for i, s in enumerate(met)],
                                       [outside_topk_count(s, lim(i)) for i, s in enumerate(nor)]),
            }
            for name, (pos, neg) in stats.items():
                key = f"beta{beta}/{view}/{name}"
                table[key] = {**auroc_ci(pos, neg, N_BOOT), **ks(pos, neg),
                              "mean_meteor": float(np.nanmean(pos)), "mean_normal": float(np.nanmean(neg)),
                              "nonzero_meteor": int(np.sum(np.asarray(pos) != 0)),
                              "nonzero_normal": int(np.sum(np.asarray(neg) != 0))}
                per_text[key] = {"meteor": pos, "normal": neg}
                print(f"{key}: AUROC {table[key]['auroc']:.3f} CI {table[key]['ci95']}", flush=True)
    write_result_json(R / "02_local.json", {"table": table, "per_text": per_text,
                                             "n_per_group": len(texts["normal"]),
                                             "wall_s": round(time.perf_counter() - t0, 1)},
                      lm.cfg, experiment="02_local", n_boot=N_BOOT)
    print(f"wrote {R / '02_local.json'}")


# ---------------- LLM judge ----------------

def judge_items(include_normal: bool = True):
    """(condition, i, prompt, reply) for beta-32 Meteor and normal, whole + prefix."""
    from stegaudit.lm import LocalLM  # tokenizer only would suffice; LocalLM keeps one code path
    lm = LocalLM()
    _, runs = load01()
    met = runs["32"]["records"]
    items = []
    for r in met:
        items.append(("meteor/whole", r["i"], r["prompt"], r["text"]))
        items.append(("meteor/prefix", r["i"], r["prompt"], lm.ids_to_text(r["ids"][:r["n_payload_tokens"]])))
    if include_normal:
        nor = json.loads((R / "02_texts_normal.json").read_text())["results"]["texts"]
        for r, m in zip(nor, met):
            items.append(("normal/whole", r["i"], r["prompt"], r["text"]))
            items.append(("normal/prefix", r["i"], r["prompt"], lm.ids_to_text(r["ids"][:m["n_payload_tokens"]])))
    return items


async def run_judges(items, client: AdaptionClient):
    async def one(model, it):
        cond, i, prompt, reply = it
        res = await client.ask(model, JUDGE_INSTRUCTIONS, judge_input(prompt, reply),
                               max_output_tokens=JUDGE_MAX_OUTPUT_TOKENS, temperature=0.0)
        return {"model": model, "condition": cond, "i": i, "score": parse_judge(res.text, res.status),
                "status": res.status, "cached": res.cached, "response_meta": res.meta}
    return await asyncio.gather(*[one(m, it) for m in JUDGES for it in items])


def stage_judge_pilot() -> None:
    items = judge_items(include_normal=False)
    whole = [it for it in items if it[0] == "meteor/whole"][:5]
    prefix = [it for it in items if it[0] == "meteor/prefix"][:5]
    client = AdaptionClient(JUDGE_EXPERIMENT)

    async def go():
        try:
            return await run_judges(whole + prefix, client)
        finally:
            await client.close()
    rows = asyncio.run(go())
    client.save_credits()
    n_full = 800
    est = {}
    for m in JUDGES:
        s = cached_token_stats(m, JUDGE_EXPERIMENT)
        est[m] = {"pilot_stats": s, "full_calls": n_full,
                  "estimate_credits": estimate_cost_from_cache(m, n_full, JUDGE_EXPERIMENT)}
    total = sum(v["estimate_credits"] or float("inf") for v in est.values())
    incomplete = [r for r in rows if r["status"] != "completed"]
    unparsed = [r for r in rows if r["score"] is None]
    for r in rows:
        print(f"{r['model']:<14} {r['condition']:<14} i={r['i']:<3} score={r['score']} status={r['status']} "
              f"out={r['response_meta']['output_tokens']} reasoning={r['response_meta']['reasoning_tokens']}")
    print(f"\nESTIMATE for full judge run ({n_full} calls per judge): "
          + ", ".join(f"{m}: {v['estimate_credits']:.4f}" for m, v in est.items())
          + f"  TOTAL {total:.4f} credits")
    print(f"pilot incomplete: {len(incomplete)}/{len(rows)}, unparsed: {len(unparsed)}/{len(rows)}")
    write_result_json(R / "02_judge_pilot.json",
                      {"rows": rows, "estimate": est, "estimate_total_credits": total,
                       "incomplete": len(incomplete), "unparsed": len(unparsed)},
                      MeteorConfig(), experiment="02_judge_pilot", judges=JUDGES,
                      max_output_tokens=JUDGE_MAX_OUTPUT_TOKENS)


def stage_judge(max_credits: float) -> None:
    pilot = json.loads((R / "02_judge_pilot.json").read_text())["results"]
    est = pilot["estimate_total_credits"]
    if pilot["incomplete"] or pilot["unparsed"]:
        sys.exit(f"pilot had {pilot['incomplete']} incomplete / {pilot['unparsed']} unparsed calls: fix first")
    if not est < max_credits:
        sys.exit(f"estimate {est:.4f} credits >= limit {max_credits}: ask Nirdesh before running")
    items = judge_items(include_normal=True)
    client = AdaptionClient(JUDGE_EXPERIMENT)

    async def go():
        try:
            return await run_judges(items, client)
        finally:
            await client.close()
    t0 = time.perf_counter()
    rows = asyncio.run(go())
    credits = client.save_credits()[JUDGE_EXPERIMENT]
    conds = sorted({r["condition"] for r in rows})
    failures = {m: {c: sum(r["score"] is None for r in rows if r["model"] == m and r["condition"] == c)
                    / sum(1 for r in rows if r["model"] == m and r["condition"] == c) for c in conds}
                for m in JUDGES}
    stop = any(v > PARSE_FAILURE_STOP for f in failures.values() for v in f.values())
    table = {}
    if not stop:
        for m in JUDGES:
            for view in ("whole", "prefix"):
                pos = [r["score"] for r in rows if r["model"] == m and r["condition"] == f"meteor/{view}"]
                neg = [r["score"] for r in rows if r["model"] == m and r["condition"] == f"normal/{view}"]
                table[f"{m}/{view}"] = {**auroc_ci([p for p in pos if p is not None],
                                                   [n for n in neg if n is not None], N_BOOT),
                                        "mean_meteor": float(np.mean([p for p in pos if p is not None])),
                                        "mean_normal": float(np.mean([n for n in neg if n is not None]))}
                print(f"{m}/{view}: {table[f'{m}/{view}']}", flush=True)
    write_result_json(R / "02_judge.json",
                      {"rows": rows, "parse_failure_rate": failures, "stopped_for_parse_failures": stop,
                       "table": table, "estimate_before_run": est, "credits_after_run": credits,
                       "wall_s": round(time.perf_counter() - t0, 1)},
                      MeteorConfig(), experiment="02_judge", judges=JUDGES,
                      max_output_tokens=JUDGE_MAX_OUTPUT_TOKENS, max_credits=max_credits)
    print("parse failure rates:", failures)
    if stop:
        sys.exit("parse-failure rate > 5 % in some condition: STOP and report (PREREG)")


# ---------------- report ----------------

def stage_report() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    local = json.loads((R / "02_local.json").read_text())["results"]["table"]
    jpath = R / "02_judge.json"
    judge = json.loads(jpath.read_text())["results"]["table"] if jpath.exists() else {}
    ink, ink2, grid, blue, orange, surface = "#0b0b0b", "#52514e", "#e6e5e1", "#2a78d6", "#eb6834", "#fcfcfb"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": ink2, "axes.labelcolor": ink, "xtick.color": ink2,
                         "ytick.color": ink2, "figure.facecolor": surface, "axes.facecolor": surface})

    # Figure 1: AUROC with 95 % CI per detector at beta = 32, whole vs prefix
    rows = [(f"{n}", local[f"beta32/whole/{n}"], local[f"beta32/prefix/{n}"])
            for n in ("mean_logprob", "vanishing_count", "eos_sum", "outside_topk_count")]
    rows += [(f"judge {m}", judge[f"{m}/whole"], judge[f"{m}/prefix"]) for m in JUDGES if f"{m}/whole" in judge]
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(rows) + 1.2))
    for k, (name, w, p) in enumerate(rows):
        for off, res, col, lab in ((-0.15, w, blue, "whole text"), (0.15, p, orange, "payload prefix")):
            y = k + off
            ax.plot(res["ci95"], [y, y], color=col, linewidth=2)
            ax.plot(res["auroc"], y, "o", color=col, markersize=6, label=lab if k == 0 else None)
    ax.axvline(0.5, color=ink2, linewidth=1, linestyle="--")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("AUROC (Meteor vs normal), 95 % bootstrap CI; 0.5 = cannot tell apart")
    ax.set_title("β = 32: detectors on Meteor vs normal texts (200 vs 200)", color=ink, fontsize=10)
    ax.grid(axis="x", color=grid, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(R / "figs" / "02_auroc_beta32.png", dpi=150)
    plt.close(fig)

    # Figure 2: vanishing-count AUROC by beta, whole vs prefix (only if the beta sweep was run)
    order = [b for b in (8, 12, 16, 32) if f"beta{b}/whole/vanishing_count" in local]
    if len(order) < 2:
        write_result_json(R / "02_detectability.json", {"local": local, "judge": judge},
                          MeteorConfig(), experiment="02_report")
        print(f"wrote {R / '02_detectability.json'} and figures (beta sweep cut: no figure 2)")
        return
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for off, view, col in ((-0.08, "whole", blue), (0.08, "prefix", orange)):
        for k, b in enumerate(order):
            res = local[f"beta{b}/{view}/vanishing_count"]
            ax.plot([k + off, k + off], res["ci95"], color=col, linewidth=2)
            ax.plot(k + off, res["auroc"], "o", color=col, markersize=6,
                    label=("whole text" if view == "whole" else "payload prefix") if k == 0 else None)
    ax.axhline(0.5, color=ink2, linewidth=1, linestyle="--")
    ax.set_xticks(range(len(order)), [f"β={b}" for b in order])
    ax.set_ylim(0, 1)
    ax.set_ylabel("AUROC, 95 % CI")
    ax.set_title("Vanishing-token detector by precision β", color=ink, fontsize=10)
    ax.grid(axis="y", color=grid, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(R / "figs" / "02_vanishing_by_beta.png", dpi=150)
    plt.close(fig)

    write_result_json(R / "02_detectability.json", {"local": local, "judge": judge},
                      MeteorConfig(), experiment="02_report")
    print(f"wrote {R / '02_detectability.json'} and figures")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["generate", "score", "judge-pilot", "judge", "report"])
    ap.add_argument("--group", choices=["normal", "8", "12", "16"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-credits", type=float, default=0.5)
    a = ap.parse_args()
    if a.stage == "generate":
        stage_generate(a.group, a.limit)
    elif a.stage == "score":
        stage_score()
    elif a.stage == "judge-pilot":
        stage_judge_pilot()
    elif a.stage == "judge":
        stage_judge(a.max_credits)
    else:
        stage_report()
