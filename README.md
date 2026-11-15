# StegAudit

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
- `pytest -q`: everything, about 6–7 min (132 tests on 2026-11-15). Most of the time is `tests/test_meteor_qwen.py`: 50 real Meteor round trips, decoded from ids and from text.
- `pytest -q -m "not slow"`: skips only the real-model Meteor round trips, about 40 s. Use it for quick checks during live edits, then run the full suite.

The first run also includes the model download. Remote tests use a fake SDK client: no network, no credits.

## Experiments

| Script | What it does | Remote calls | CPU time |
|---|---|---|---|
| `python experiments/00_adaption_check.py [--max-output-tokens N]` | Lists hosted Adaption models; one tiny chat call; prints its cost | 1 (cached afterwards) | seconds |

Every result JSON has a `meta` block: git commit and dirty flag, seed, local model and revision, config, library versions, timestamp. Remote answers are cached in `cache/` (gitignored); rerunning a script reads the cache and costs nothing.

More experiments are added here as they land.

## Future work (cut for the course deadline, 2026-11-15)
These were planned but cut to meet the deadline. None of them has been started.
- **RQ2: paraphrasing** as a defence (does a paraphraser between Sender and Receiver destroy the channel, and at what cost?).
- **Shimmer** and other stego schemes, as a comparison to Meteor.
- **RRNs-HT** transcript detector (to be implemented only from the paper).
- **Qwen2.5-1.5B-Instruct** as the channel model (its revision is already pinned in `config.py`).
- **β sweep for the detectors** in experiment 02 (β = 8, 12, 16). Experiment 01 already measured how the vanishing rate depends on β.
- **Further efficiency analysis** beyond the exploratory 01c result (about 0.58 bits per bit of top-k entropy at β = 32).
- Retokenisation-robust encoding (Nozaki & Murawaki 2022; SyncPool, Qi et al. 2024). Not needed so far: the PREREG kill rule did not fire.
