# References

Each entry is marked **[VERIFIED]** (Nirdesh checked it against the PDF), **[VERIFY]** (written from memory, not yet checked against the PDF) or **TODO** (incomplete on purpose).

## Papers
- **[VERIFIED]** Anwar, U., Piskorz, J., Baek, D. D., Africa, D., Weatherall, J., Tegmark, M., Schroeder de Witt, C., van der Schaar, M., Krueger, D. (2026). *A Decision-Theoretic Formalisation of Steganography With Applications to LLM Monitoring.* arXiv:2602.23163.
  Used for: the steganographic gap in Phase 2. The normalised gap is implemented only after checking its definition against this paper.
- **[VERIFIED]** Motwani, S. R., Baranchuk, M., Strohmeier, M., Bolina, V., Torr, P. H. S., Hammond, L., Schroeder de Witt, C. (2024). *Secret Collusion among AI Agents: Multi-Agent Deception via Steganography.* NeurIPS 2024. arXiv:2402.07510.
  Used for: threat model.
- **[VERIFY]** Kaptchuk, G., Jois, T. M., Green, M., Rubin, A. D. (2021). *Meteor: Cryptographically Secure Steganography for Realistic Distributions.* ACM CCS 2021.
  Used for: the encode/decode scheme in `meteor.py`.
- **[VERIFY]** Blocki, J., Christin, N., Datta, A., Procaccia, A. D., Sinha, A. (2013). *Audit Games.* IJCAI 2013. arXiv:1303.0356.
  Used for: Phase 3 audit LPs.
- **[VERIFY]** Kiekintveld, C., Jain, M., Tsai, J., Pita, J., Ordóñez, F., Tambe, M. (2009). *Computing Optimal Randomized Resource Allocations for Massive Security Games.* AAMAS 2009.
  Used for: the M2 single-target security game in Phase 3.
- **[VERIFY]** Nozaki, J., Murawaki, Y. (2022). Tokenization ambiguity in linguistic steganography (prefix-removal disambiguation). AACL-IJCNLP 2022. Exact title, authors and page numbers to be checked against the PDF.
  Used for: candidate fix for retokenisation mismatches, only if the kill rule in PREREG.md (P1d) fires. Reported to remove ambiguous tokens from the candidate set, which changes the distribution; check against the paper whether this breaks provable security.
- **[VERIFY]** Qi et al. (2024). *SyncPool* (tokenization-ambiguity handling for provably secure steganography). Exact title, venue and authors to be checked against the PDF.
  Used for: second candidate fix for retokenisation mismatches; claims to preserve security. Not implemented; only if the P1d kill rule fires.
- **TODO** *Breaking the Provable Security* (ACL Findings 2026): authors and full title not yet recorded.
  Used for: the low-probability vanishing phenomenon in `quantize.py`.

## Models
- Qwen2.5-0.5B-Instruct (`Qwen/Qwen2.5-0.5B-Instruct`), Hugging Face, local stego channel (default).
- Qwen2.5-1.5B-Instruct (`Qwen/Qwen2.5-1.5B-Instruct`), optional local channel.
- Adaption-hosted models (Sentinel / Receiver answering / LLM judge): exact ids go in `results/adaption_models.json` after the list-models call.

## Datasets
- MMLU (`cais/mmlu`, Hugging Face `datasets`), Phase 2.

## Libraries (exact versions in `requirements.txt`)
PyTorch 2.15.0 (CPU), Hugging Face transformers 5.23.0 / tokenizers 0.23.3, cryptography 50.0.2 (X25519, HKDF, HMAC), NumPy, SciPy (`linprog`, HiGHS), pandas, matplotlib, scikit-learn, datasets, pytest, Adaption Python SDK 0.19.0 (`adaption`), python-dotenv.
