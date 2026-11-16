# StegAudit

Repository: https://github.com/nirdesho6o/StegAudit

Meteor steganography pilot (local Qwen model, CPU) and stake-aware audit LPs.
CNS course project, NIT Warangal. See `CLAUDE.md` for the full spec and `AI_USE.md` for what was AI-generated.

## Install (CPU only, Python ≥ 3.11)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # pulls the CPU-only torch wheel
pip install -e .
```

Remote models (Adaption): put `ADAPTION_API_KEY=...` in `.env`. The file is gitignored and is never committed.

The first model load downloads Qwen2.5-0.5B-Instruct (about 1 GB) at the pinned revision into the Hugging Face cache, outside the repo.

## Tests

```bash
pytest -q
```

Runtime on CPU (one thread, by design), once the model is cached:
- `pytest -q`: everything, about 7 min in one process (330 tests on 2026-11-16; peak memory about 5.7 GB). Most of the time is `tests/test_meteor_qwen.py`: 50 real Meteor round trips, decoded from ids and from text.
- `pytest -q -m "not slow"`: skips only the real-model Meteor round trips, about 40 s. Use it for quick checks during live edits, then run the full suite.

The first run also includes the model download. Remote tests use a fake SDK client: no network, no credits.

## Experiments

Run from the repo root with the venv active. Times are measured on this machine (CPU, 1 thread per process).

| Script | What it does | Remote calls (credits) | Time |
|---|---|---|---|
| `experiments/00_adaption_check.py [--max-output-tokens N]` | Lists hosted Adaption models; one tiny chat call; prints its cost | 1 | seconds |
| `experiments/01_meteor_roundtrip.py` | 200 Meteor messages at β = 32 (full texts) + β = 8/12/16 (payload only); round trip, capacity, vanishing, EOS masking | 0 | 88 min |
| `experiments/01b_redecode.py` | Fresh-process re-decode of all 800 messages from disk (independent check of 01) | 0 | 58 min |
| `experiments/01c_entropy_explore.py` | *Exploratory:* per-step entropy and bits/entropy on 30 messages per β | 0 | ~4 min |
| `experiments/01d_followups.py` | *Exploratory:* top-1 p at cover-vanishing steps; payload-only retokenisation rate | 0 | ~2 min |
| `experiments/02_detectability.py generate --group normal` | 200 normal (non-stego) samples | 0 | 37 min |
| `experiments/02_detectability.py score` | Local detectors (log-prob + KS, vanishing, EOS, outside top-k), whole text + prefix | 0 | 12 min |
| `experiments/02_detectability.py judge-pilot` | 10 LLM-judge calls + cost estimate | 10 (~0.003) | ~2 min |
| `experiments/02_detectability.py judge [--max-credits 0.5]` | LLM judge (deepseek-v4-flash) on 400 whole texts; refuses to run if the pilot estimate exceeds the limit | 400 (~0.10) | ~20 min |
| `experiments/02_detectability.py report` | Summary JSON + AUROC figure | 0 | seconds |
| `experiments/02b_prefix_check.py` | *Exploratory:* is the prefix log-prob signal a payload-boundary artefact? | 0 | 5 min |
| `experiments/04_audit_lp.py [--phase2] [--quick]` | Audit LPs vs colluders on synthetic stakes; P3a audits-vs-ε; `--phase2` adds both (d, f) pairs from `results/d_estimate.json` | 0 | ~25–34 min (estimated) |

Rule for this project: any single run expected to take more than 45 minutes is announced and approved first. 01 and 01b ran before that rule existed.

Every result JSON has a `meta` block: git commit and dirty flag, seed, local model and revision, config, library versions, timestamp. Remote answers are cached in `cache/` (gitignored); rerunning a script reads the cache and costs nothing.

## Report

- `report/report.tex` → `report/report.pdf` (20 pages). Build: `cd report && pdflatex report.tex && pdflatex report.tex` (needs TeX Live with `tcolorbox`, `animate`, TikZ, Latin Modern; e.g. `texlive-latex-extra texlive-pictures`).
- `report/figs/`: copies of `results/figs/app_tab1–4.png`, `02_auroc_beta32.png` (as `02_auroc.png`) and `04_undetected_vs_gini.png` (as `04_undetected.png`).
- The animation in Section 4.4 plays in Adobe Acrobat Reader or Okular; other viewers show its first frame, and the storyboard figure shows every frame.
- `docs/stegaudit_explainer.html`: the same walkthrough as an interactive page; open it in any browser.
- Every number in the report was checked against `results/LOG.md` and `results/*.json` (2026-11-16).

## Future work (cut for the course deadline, 2026-11-15)
These were planned but cut to meet the deadline. None of them has been started.
- **RQ2: paraphrasing** as a defence (does a paraphraser between Sender and Receiver destroy the channel, and at what cost?).
- **Shimmer** and other stego schemes, as a comparison to Meteor.
- **RRNs-HT** transcript detector (to be implemented only from the paper).
- **Qwen2.5-1.5B-Instruct** as the channel model (its revision is already pinned in `config.py`).
- **β sweep for the detectors** in experiment 02 (β = 8, 12, 16). Experiment 01 already measured how the vanishing rate depends on β.
- **Further efficiency analysis** beyond the exploratory 01c result (about 0.58 bits per bit of top-k entropy at β = 32).
- Retokenisation-robust encoding (Nozaki & Murawaki 2022; SyncPool, Qi et al. 2024). Not needed so far: the PREREG kill rule did not fire.
