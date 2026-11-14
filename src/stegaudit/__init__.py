"""StegAudit: a Meteor steganography pilot and stake-aware audit LPs.

Idea in plain words
-------------------
Two colluding AI agents (Sender, Receiver) share a secret key. The Sender
writes normal-looking text with a language model, but picks each token using
the key so that a hidden payload rides along (Meteor). Without the key the
text looks like an ordinary model sample, so reading transcripts cannot catch
it. The research question is where an auditor with a limited budget should
look instead: at the high-stakes *decisions* the colluders influence.

Modules (filled in phase by phase):
  config    - every tunable parameter in one place (beta, top-k, model, ...)
  keys      - X25519 + HKDF-SHA256 -> stego keys
  prg       - keyed pseudorandom mask bits
  quantize  - float probabilities -> integer intervals at beta bits
  lm        - local model wrapper (next-token distribution)
  meteor    - encode / decode
  remote    - Adaption client for Sentinel / Receiver / LLM judge
  detect, gap, audit_lp, simulate - later phases

Security assumption for the whole package: the PRG is secure, the key is
secret, and Sender and Receiver compute bit-identical model distributions.
If the last one fails, decoding breaks (correctness), not secrecy.
"""

__version__ = "0.1.0"
