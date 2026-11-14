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

## Tests

```bash
pytest -q
```

## Experiments

TODO: filled in as each experiment lands (commands and expected CPU runtimes).
