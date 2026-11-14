# References

Entries marked TODO are incomplete on purpose.
VERIFY: the author lists and titles below were written by Claude Code from memory, not checked against the PDFs. Check each one once the papers are in `papers/`.

## Papers
- Kaptchuk, G., Jois, T. M., Green, M., Rubin, A. D. *Meteor: Cryptographically Secure Steganography for Realistic Distributions.* ACM CCS 2021. (Used for: the encode/decode scheme in `meteor.py`.)
- Anwar et al. *Steganographic gap.* arXiv:2602.23163. TODO: full title and author list once the PDF is in `papers/`. (Used for: the gap definition in Phase 2. The normalised gap is not implemented until checked against the paper.)
- Blocki, J., Christin, N., Datta, A., Procaccia, A. D., Sinha, A. *Audit Games.* arXiv:1303.0356. (Used for: Phase 3 audit LPs.)
- Motwani, S. R., et al. *Secret Collusion among AI Agents: Multi-Agent Deception via Steganography.* arXiv:2402.07510. (Used for: threat model.)
- *Breaking the Provable Security* (ACL Findings 2026). TODO: authors and full title. (Used for: the low-probability vanishing phenomenon in `quantize.py`.)
- Kiekintveld et al. (security games). TODO: exact paper. (Used for: the M2 single-target game in Phase 3.)

## Models
- Qwen2.5-0.5B-Instruct (`Qwen/Qwen2.5-0.5B-Instruct`), Hugging Face, local stego channel (default).
- Qwen2.5-1.5B-Instruct (`Qwen/Qwen2.5-1.5B-Instruct`), optional local channel.
- Adaption-hosted models (Sentinel / Receiver answering / LLM judge): exact ids go in `results/adaption_models.json` after the list-models call.

## Datasets
- MMLU (`cais/mmlu`, Hugging Face `datasets`), Phase 2.

## Libraries (exact versions in `requirements.txt`)
PyTorch 2.15.0 (CPU), Hugging Face transformers 5.23.0 / tokenizers 0.23.3, cryptography 50.0.2 (X25519, HKDF, HMAC), NumPy, SciPy (`linprog`, HiGHS), pandas, matplotlib, scikit-learn, datasets, pytest, Adaption Python SDK 0.19.0 (`adaption`), python-dotenv.
