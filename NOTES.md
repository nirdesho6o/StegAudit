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
