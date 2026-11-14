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

Runtime on CPU: about 10–15 s once the model is cached (the model is loaded twice plus once in a fresh process for the determinism tests). The first run also includes the model download. Remote tests use a fake SDK client: no network, no credits.

## Experiments

| Script | What it does | Remote calls | CPU time |
|---|---|---|---|
| `python experiments/00_adaption_check.py [--max-output-tokens N]` | Lists hosted Adaption models; one tiny chat call; prints its cost | 1 (cached afterwards) | seconds |

Every result JSON has a `meta` block: git commit and dirty flag, seed, local model and revision, config, library versions, timestamp. Remote answers are cached in `cache/` (gitignored); rerunning a script reads the cache and costs nothing.

More experiments are added here as they land.
