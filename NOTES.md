# Notes

## Viva drills
Questions only. Nirdesh answers them himself.

### Phase 0: setup
1. Why must the Meteor channel run on a local model and never on a hosted API? Name two concrete things that would break if it ran on Adaption.
2. `lm.py` sets one CPU thread and `torch.use_deterministic_algorithms(True)`. How can more threads change the logits, and why does a difference in the last bit of one probability matter for Meteor decoding?
3. Why is the Qwen model loaded by a commit hash instead of by its name? Give a scenario where loading by name gives a wrong experimental result without any error.
4. A hosted model at temperature 0 can still give different answers to the same prompt. Why? How does this project make remote answers reproducible anyway, and what must be said in the report if `cache/` is deleted and the experiment rerun?
5. The first test call (16-token cap) returned `status=incomplete` with empty text, and the second (256 cap) returned "ok" but billed 24 output tokens. Explain both. Why would dropping incomplete answers in Phase 2 bias the steganographic gap?
