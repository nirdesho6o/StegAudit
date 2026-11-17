# Notes

## Viva drills
Questions only. Nirdesh answers them himself.

### Phase 0: setup
1. Why must the Meteor channel run on a local model and never on a hosted API? Name two concrete things that would break if it ran on Adaption.
2. `lm.py` sets one CPU thread and `torch.use_deterministic_algorithms(True)`. How can more threads change the logits, and why does a difference in the last bit of one probability matter for Meteor decoding?
3. Why is the Qwen model loaded by a commit hash instead of by its name? Give a scenario where loading by name gives a wrong experimental result without any error.
4. A hosted model at temperature 0 can still give different answers to the same prompt. Why? How does this project make remote answers reproducible anyway, and what must be said in the report if `cache/` is deleted and the experiment rerun?
5. The first test call (16-token cap) returned `status=incomplete` with empty text, and the second (256 cap) returned "ok" but billed 24 output tokens. Explain both. Why would dropping incomplete answers in Phase 2 bias the steganographic gap?

### Phase 1a: keys.py, prg.py, quantize.py
**keys.py**
1. Walk through what Sender and Receiver each compute in X25519. Which values are public, which are secret, and why can an eavesdropper who sees both public keys not compute the shared secret?
2. Why is the raw X25519 output passed through HKDF instead of being used directly as the key? Why are separate `mask_key` and `tag_key` derived from `stego_key`, and what is the `info` string for?
3. `Party.from_seed` makes the private key from a hash of a seed. Why does it exist, and why would using it in a real deployment destroy secrecy? Why does the code reject an all-zero shared secret?
4. In the threat model, why must the key exchange happen before deployment rather than over the monitored channel?

**prg.py**
5. Explain the PRG as a "regenerable one-time pad". What does the Receiver need to know to regenerate the same mask as the Sender?
6. Why are msg_id and step encoded as fixed 8-byte fields instead of being joined as strings? Give a concrete collision that the fixed-width encoding prevents.
7. Suppose two messages are sent with the same key and the same msg_id. What can a monitor learn, (a) if the two payloads are equal, and (b) if they differ?
8. What security property of HMAC-SHA256 does the whole construction rely on? Why are "payload" and "cover" separate labels?

**quantize.py**
9. Why must the probabilities be turned into integers at all? What would go wrong if Sender and Receiver compared floats directly?
10. With `width_i = floor(p_i · 2^β)` and the remainder given to the argmax, show that each non-argmax token's sampling probability is within 2^-β of p_i. What is the bound on the total variation distance for k tokens?
11. What is "low-probability vanishing"? Show that a token vanishes exactly when p_i < 2^-β. Why did we expect vanishing to be almost absent at β = 32 with top-k = 50, and why does that make the β sweep necessary?
12. Why are the probabilities computed in float64 even though the model runs in fp32? What would happen at β = 32 with fp32 probabilities?
13. Why can `find(x)` never return a vanished token? Change β to 16 live: which tests would you expect to still pass, and what changes in the vanishing count?

### Phase 1b: lm.py step API, meteor.py
**lm.py step API**
1. What is the KV cache, and why does using it make each generated token cheaper? The KV-cache path and a full recompute differed in 146,520 of 151,936 logits (max 2.0e-5). Why does that force Sender and Receiver to use the same path, even though the difference is tiny?
2. Why are the top-k tokens sorted by token id rather than by probability? What would go wrong if the Sender sorted one way and the Receiver the other?
3. EOS is masked by setting its logit to -inf *before* top-k. What would change if it were removed *after* top-k instead? What does `eos_mass` measure, and why is it logged?

**meteor.py**
4. Run the worked example yourself: p = [0.5, 0.3, 0.2], β = 4, message bits 1011, mask 0110. Give the widths, the intervals, x, the chosen token, the common prefix, and the recovered bits. Then do it again with mask 0011 and explain why 0 bits are sent.
5. Why is the number of bits sent at a step equal to the common prefix length of lo and hi−1? Why can the Receiver compute this number without knowing the key?
6. Why is the token Meteor emits a fair sample from the (quantised) model distribution, whatever the message? Which test checks this exactly, and how?
7. What is the HMAC tag for? With a t-bit tag, what fraction of wrong-key decodes should pass? The Qwen test accepted 2 of 300 wrong keys with t = 8: is that consistent? Why does the tag include msg_id?
8. When fewer than β payload bits remain, the chunk is padded with zeros. Why does this not make the token choice biased?
9. Explain the retokenisation failure on msg 29 (`Program` + `ming` vs `Programming`). Why is it a BPE problem and not a Meteor problem? Why is it important that the failure is detected rather than producing a wrong message?
10. EOS masking during the payload is a deviation from pure Meteor. Why was it needed? How could a detector exploit it, and what does `eos_masked_steps` let you measure?
11. Live modification: change `tag_bits` from 8 to 4. What happens to the wrong-key test, and why? Change β from 32 to 8: what happens to bits per token and to the vanishing count?

### Phase 1c: detect.py and experiment 02 (detectability)
1. What does AUROC measure for a detector, and why is 0.5 the value Meteor's security claim predicts? Why is an AUROC clearly *below* 0.5 also evidence of a leak?
2. The log-prob detector uses the full-vocabulary log-probability, not the top-k one. What would go wrong with the top-k version on Meteor's payload tokens (hint: EOS masking)?
3. The vanishing-token detector found zero events in both groups at β = 32. Why was that expected, and why does it mean the detector "passes" trivially rather than showing Meteor is safe?
4. On the payload prefix, mean log-prob gave AUROC 0.447, but dropping the last prefix token gave 0.524 and the last token alone gave 0.169. Explain why the token where the payload ends tends to be improbable. Why is the "prefix" view not a realistic detector?
5. The monitor scores texts with one full forward pass, while the Receiver must use the KV-cache path. Why may the monitor use a different (slightly different) computation, but the Receiver may not?
6. Why was the LLM judge reduced to one model on whole texts only? What limitation does that leave, and why is it acceptable here?
7. Why must a parse failure (incomplete or malformed judge answer) be counted and reported, rather than dropped?
