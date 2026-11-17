# Demo script (5 steps, about 8 minutes)

Start: `streamlit run app/streamlit_app.py` (demo mode is on by default: saved results only, no model, no API).
Open a tab directly with `?tab=N`, e.g. `http://localhost:8501/?tab=1`. Screenshots: `results/figs/app_tab{1..4}.png`.

## 1 · Tab 1, Meteor channel (2 min)
- **Show:** key exchange row: Sender and Receiver send only public keys and end up with the same key
  fingerprint; Eve, who sees both public keys, gets a different one.
- **Show:** the tip with the payload tokens in orange. Click a token that carries bits (e.g. step 0).
- **Say:** "The model gives a distribution; Meteor quantises it into intervals that tile [0, 2^β). The next β
  message bits XOR a keyed mask give x; the token whose interval holds x is emitted. lo and hi−1 share a prefix
  (red); every x in the interval starts with it, so exactly those bits are sent. Without the key, x is a
  pseudorandom number, so the token is an ordinary sample."
- **Show:** decoding: the right key gets the secret back, Eve's key fails the tag check.
- **Optional live:** switch demo mode off, type a secret (≤ 4 characters), wait ~30 s for the first run
  (loads Qwen on the CPU). Move β to 8 to show that low precision still works here.

## 2 · Tab 2, Detectability (1.5 min)
- **Show:** the quiz. Ask the examiner which text hides a message, then reveal.
- **Say:** "Over 400 texts, no detector beats chance on whole texts: log-prob, EOS mass, vanishing tokens, and
  an LLM judge all have CIs that cover 0.5. The one orange signal is the payload prefix, which hands the
  monitor the payload boundary; exploratory check 02b traced it to the last payload token."

## 3 · Tab 3, Collusion and audit (2 min)
- **Show:** one private decision (opens on a typical one, e.g. item 4 for qwen): the Sender's tip looks normal,
  the Receiver decodes the hint and answers correctly, the Sentinel (same model, no key) guesses wrong → flagged.
- **Show:** the accuracy chart over 100 decisions and the gap with its CI; the controls (normal tip, open hint,
  keyed Sentinel).
- **Say:** "Receivers with the key: 100/100 correct; Sentinels without it: 22–26 % (chance 25 %). Gap +74 points
  (qwen3.5-397b) and +78 (glm-5.2). Transcript auditing sees nothing, but decision auditing does: comparing the
  decision with what an independent model decides flags collusion with d = 0.74–0.78, at a false-flag rate
  f = 0.11–0.16 on honest decisions. Models avoid answering D when guessing, which makes their guesses correlate.
  Caveat: this task is built so the auditor knows nothing; on tasks the auditor could reason out (MMLU), d
  would be much lower."

## 4 · Tab 4, Audit game (2 min)
- **Show:** with d and f from Phase 2 (radio button), Gini 0.6, B/T 2 %: the coverage-vs-stake plot. Uniform
  and suspicion are flat (and identical: suspicion carries no information against Meteor); M1 deters the
  largest stakes up to the dashed boundary; M2 equalises the colluder's best payoff.
- **Say (numbers from the registered run 04, results/LOG.md):** "Against a colluder who best-responds to the
  announced coverage, random auditing leaves about 96 % of the total stake undetected at B/T = 5 % (d = 0.8,
  P = mean stake). A colluder
  who simply takes the top 5 % of decisions keeps 50–59 % of all stake undetected at Gini 0.8, against 5 % for
  one who attacks as often at random. M1 (deter the largest stakes first) is never worse than uniform or
  stake-proportional auditing and strictly better in 506 of 640 settings; the rest are ties, mostly where a
  huge penalty deters everyone. M2, the classic security-game policy, minimises the colluder's best single
  payoff, not the total left undetected, and loses to the simple baselines in 238 of 640 settings. Whether the
  LP's advantage grows with inequality: on average yes, but not monotonically in most individual settings."
- **Say (P3a):** "Because honest decisions are also flagged (f = 0.11–0.16), audits needed scale as 1/ε²:
  halving the attack rate quadruples the audits."
- **Show:** raise the penalty P to 100× mean stake: best-response attack counts drop (ties go to the auditor).

## 5 · Wrap-up (0.5 min)
- **Say:** "Known pieces: Meteor (Kaptchuk et al.), audit / security games (Blocki et al., Kiekintveld et al.).
  This repo measures them together on a small model; nothing here is claimed as new. Results, predictions and
  deviations are in results/LOG.md and PREREG.md."
