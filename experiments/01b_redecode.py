"""Experiment 01b: independent re-decode of experiment 01 in a FRESH process.

Outside judge for experiment 01. Reads only results/01_meteor_roundtrip.json
from disk: no state is shared with the process that encoded. The Receiver's
keys are re-derived from the logged seed (pair_from_seed(seed)[1]).

For EVERY message it recomputes both decode paths:
  * text path: decode(text), retokenising the saved text;
  * ids path:  decode_ids(saved ids).
and compares each with the original message bits AND with what experiment 01
recorded. Any disagreement is listed.

Output: results/01b_redecode.json
Usage:  python experiments/01b_redecode.py [--in PATH] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from stegaudit.config import MeteorConfig
from stegaudit.keys import pair_from_seed
from stegaudit.lm import LocalLM
from stegaudit.meteor import decode, decode_ids
from stegaudit.provenance import REPO_ROOT, write_result_json


def main(inp: Path, out: Path) -> None:
    data = json.loads(inp.read_text())
    meta, runs = data["meta"], data["results"]["runs"]
    _, keys_r = pair_from_seed(meta["params"]["key_seed"])
    lm = LocalLM()
    assert lm.cfg.model_revision == meta["model_revision"], "different model revision than experiment 01"
    beta_maxnew = {str(b): m for b, m in meta["params"]["runs_beta_maxnew"]}

    t0 = time.perf_counter()
    summary, disagreements = {}, []
    for beta, run in runs.items():
        cfg = MeteorConfig(beta=int(beta), max_new_tokens=beta_maxnew[beta])
        c = {"n": 0, "ids_ok": 0, "text_ok": 0, "text_wrong_accepted": 0, "agree_with_01": 0}
        for r in run["records"]:
            ids_res = decode_ids(lm, keys_r, r["prompt"], r["ids"], r["msg_id"], cfg)
            txt_res = decode(lm, keys_r, r["prompt"], r["text"], r["msg_id"], cfg)
            ids_ok = ids_res.ok and ids_res.message_bits == r["message"]
            txt_ok = txt_res.ok and txt_res.message_bits == r["message"]
            c["n"] += 1
            c["ids_ok"] += ids_ok
            c["text_ok"] += txt_ok
            c["text_wrong_accepted"] += txt_res.ok and txt_res.message_bits != r["message"]
            same = (ids_ok == r["ids_path_ok"]) and (txt_ok == r["text_path_ok"])
            c["agree_with_01"] += same
            if not same:
                disagreements.append({"beta": beta, "i": r["i"], "01_ids_ok": r["ids_path_ok"],
                                      "01b_ids_ok": ids_ok, "01_text_ok": r["text_path_ok"],
                                      "01b_text_ok": txt_ok})
        summary[beta] = c
        print(f"beta={beta}: {c}", flush=True)

    src = inp.resolve()
    src_s = str(src.relative_to(REPO_ROOT)) if src.is_relative_to(REPO_ROOT) else str(src)
    results = {"source": src_s, "source_git_commit": meta["git_commit"],
               "per_beta": summary, "disagreements": disagreements,
               "wall_s": round(time.perf_counter() - t0, 1)}
    write_result_json(out, results, lm.cfg, experiment="01b_redecode")
    print(f"wrote {out}; disagreements: {len(disagreements)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, default=REPO_ROOT / "results" / "01_meteor_roundtrip.json")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "01b_redecode.json")
    a = ap.parse_args()
    main(a.inp, a.out)
