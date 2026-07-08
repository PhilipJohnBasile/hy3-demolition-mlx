# Manual quality pass (#16) — done by Claude, 2026-07-08

The one step no automation could answer: not "does it pass tests" but "is it
*pleasant and capable* to actually use." Six real daily-driver / daily-agent
tasks (CSV CLI, code review, agent plan, reasoning, explanation, tool JSON),
run in production `no_think` mode, one model at a time. Outputs in
`eval/receipts/manual_pass_*.json`.

## Head-to-head: reap25 (Hy3) vs the 35B sibling

| Task | reap25 (Hy3, 7.4 tok/s) | sibling (Qwen35B, 100+ tok/s) |
|---|---|---|
| CSV stats CLI | clean argparse, helper fns | clean argparse, docstrings |
| review a buggy `average()` | caught ZeroDivision **+ TypeError**, sum(), hints — 25s | caught ZeroDivision, sum()/len(), hints — **2.3s** |
| Flask rate-limit plan | concrete plan + code | concrete plan + code |
| two-trains reasoning | correct (60t+40t=180) — 44s | correct (relative-speed) — **4.0s** |
| explain async/await | accurate + complete — 13s | *crisper*: "the scheduler is you, not the OS" — **1.1s** |
| tool-call JSON | perfect, no prose — 4s | **identical**, no prose — **0.4s** |

## Verdict

**Quality is essentially equal.** Both are correct, well-formatted, and
genuinely capable. reap25 is marginally more thorough (it flagged the extra
`TypeError`); the sibling's prose is a touch crisper. Neither is meaningfully
better to *use* — both would make a good daily model.

**Speed is decisive.** The sibling answers in 0.4–4.4 s; reap25 takes 4–44 s.
For interactive chat and especially for an agent loop (which emits thousands of
tokens per task), the sibling *feels effortless* and reap25 *feels like waiting*.

## What this means for the family

- **Daily driver + daily agent → the 35B sibling.** Equal quality, 6–10× faster.
  This is the honest surprise of the pass: the "fast small" model is the one
  you'd actually reach for all day. Its outputs are validated by deterministic
  verifiers anyway, so throughput + guardrails beat raw size.
- **reap25 (Hy3) → "the big model's answer when you want it."** A capable,
  slightly-more-thorough daily driver on a 96–128 GB Mac, but you feel the 7.4
  tok/s on long answers.
- **Streaming Hy3 → reap25 quality on any Mac (16–128 GB), slower still.** Same
  weights, so this verdict transfers; it's the run-anywhere option, not the
  interactive one.

**Bottom line:** the family is real and each tier earns its place — but for
*actual daily use*, ship the sibling as the default and keep the Hy3 tiers for
"I want the 295B's answer" / "run it on this tiny Mac." The manual pass
confirms the models are pleasant to use, not just correct on paper.
