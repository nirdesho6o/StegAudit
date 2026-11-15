"""Precompute tab 1's demo-mode example: one secret encoded at every demo beta, with the full trace.

Not an experiment (no prediction): it only saves what the app shows when the local model is not
loaded. Output: results/app_demo_meteor.json  (~1 min on CPU).
"""

from __future__ import annotations

import time

from stegaudit.appcore import DEMO_BETAS, DEMO_PROMPT, DEMO_SECRET, meteor_demo
from stegaudit.lm import LocalLM
from stegaudit.provenance import REPO_ROOT, write_result_json


def main():
    lm = LocalLM()
    t0 = time.perf_counter()
    runs = {}
    for beta in DEMO_BETAS:
        t = time.perf_counter()
        runs[str(beta)] = meteor_demo(lm, DEMO_SECRET, beta)
        runs[str(beta)]["encode_decode_s"] = round(time.perf_counter() - t, 2)
        r = runs[str(beta)]
        print(f"beta={beta}: {r['n_payload_tokens']} payload tokens, right key {r['decode_right']}, "
              f"wrong key ok={r['decode_wrong']['ok']}, {r['encode_decode_s']}s", flush=True)
    write_result_json(REPO_ROOT / "results" / "app_demo_meteor.json",
                      {"runs": runs, "wall_s": round(time.perf_counter() - t0, 1)}, lm.cfg,
                      experiment="app_demo_meteor", secret=DEMO_SECRET, prompt=DEMO_PROMPT, betas=DEMO_BETAS)


if __name__ == "__main__":
    main()
