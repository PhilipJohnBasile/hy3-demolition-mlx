# Soul-topic distillation: does it beat prompting, and does it save money?

**Date:** 2026-07-08 · **Question:** can a locally-distilled LoRA replace paid
cloud calls (Claude/Codex/OpenCode-class tools) on domain-expert ("soul topic")
work — quality *and* cost? · Held-out prompts (never used in any prior eval):
`data/soul_distill_prompts.json` (24 train + 6 eval, spanning coding, security,
math, science, design, fullstack, gamedev, legacy, music, art, perfumery).

## Method

1. **Gold targets from Sonnet** (a real paid cloud model, via two independent
   subagent calls — train set and eval set kept separate so eval is a genuinely
   fresh cloud call, not training data). Real cost: **19,967 tokens for the 6
   eval answers, 20,676 for the 24 training answers.**
2. **Train a LoRA** (rank 8, 8 layers, 200 iters) on (question → Sonnet answer)
   pairs — same recipe as the blueprint-distill experiment.
3. **Four conditions on the 6 held-out eval questions:**
   - **A** — local sibling, no prompt (the free-est option)
   - **B** — local sibling + an explicit "you are a domain expert" system prompt
   - **C** — local sibling + the distilled LoRA, no prompt
   - **D** — fresh Sonnet cloud call (the paid reference)
4. **Blind judge:** Opus (generated none of the four), reading all 4 answers
   per question with labels shuffled and hidden — scored 1–10 on accuracy /
   depth / clarity, plus a best-to-worst ranking, per question.

## Result: distillation genuinely wins among local options — first time this session

| Condition | Avg score /10 | Avg rank (1=best) |
|---|---|---|
| **D — Sonnet cloud** | **9.00** | **1.00** (won all 6/6 questions) |
| **C — distilled LoRA, no prompt** | **6.83** | **2.50** |
| B — base + expert system prompt | 6.17 | 3.17 |
| A — base, no prompt | 6.00 | 3.33 |

**C beat both A and B in 5 of 6 questions**, and did it while paying **zero**
extra prompt tokens (B pays +52 tokens/query for its system prompt; C uses the
same 27–33 tokens as the unprompted base). This is different from every prior
distillation result in this project:
- **Souls token distillation** (`docs/souls-deep-dive.md`): added nothing — base
  responded to the cue as strongly as the healed model.
- **Blueprint-stance distillation** (`docs/blueprint-ablation.md`): matched
  prompting (C ≈ B), didn't exceed it.
- **This experiment**: **C > B > A**, a real (if modest) local capability gain
  from distilling 20 gold examples.

A likely mechanism, observed directly: the LoRA **picked up Sonnet's plain-prose
style** on some questions (e.g. e02: base/prompted used 5–7 markdown
headers/bold/bullets; the distilled model matched Sonnet's near-zero markdown
density exactly). Less time spent on markdown formatting inside a fixed token
budget may mean more of the budget goes to actual content — a plausible, if
unconfirmed, reason distillation edged out both other local conditions.

## But: the gap to cloud remains large and totally consistent

**D won every single one of the 6 questions**, average score 9.0 vs the local
best (C) at 6.83 — roughly a 2-point gap on a 10-point scale, every time, no
exceptions. This matches the theme of every other benchmark in this project
(HumanEval, AIME, GPQA): **local models are honestly behind frontier cloud on
raw quality.** Distillation closes part of the gap between local options; it
does not close the gap to cloud.

## The money question: does this actually save cost?

**Real numbers, not estimates:**
- **Cloud (Sonnet), this experiment:** 40,643 tokens for 30 domain-expert Q&A
  pairs. At **current standard Sonnet pricing** ($3/M input, $15/M output —
  [Anthropic pricing page](https://platform.claude.com/docs/en/about-claude/pricing),
  checked 2026-07-08) that's roughly **$0.54 total, ~$0.018 per question**.
  At 1,000 similar questions/month: **~$18/month** in cloud API cost.
- **Local (any of A/B/C):** **$0 marginal cost** — runs on-device at 131 tok/s.
  One-time cost: ~15–20 min of GPU time to train the LoRA, done once, reused
  forever.

**Honest framing:** for *this specific kind* of task (a single, short
domain-expert question), cloud is already cheap in absolute terms — $0.018/query
is not going to break a budget by itself. The real savings show up at **volume**
(many questions/day, an agentic loop that asks many small domain questions) or
in **agentic coding sessions**, which typically burn far more tokens per
interaction than a clean Q&A pair (tool outputs, file contents, multi-turn
context) — this experiment's $0.018/query is a *floor*, not representative of a
real Claude Code / Codex-style session cost, which usually runs far higher per
interaction.

## Verdict: does this replace paying for Claude/Codex/OpenCode?

**No, not for quality-critical work — the blind judge's 9.0 vs 6.8 gap is real
and consistent.** But it's a genuine, useful finding for a specific slice of
work:

- **For repeated, similar-shaped domain questions** where "good, not best" is
  acceptable (a FAQ-style helper, a first-pass draft, a high-volume low-stakes
  loop) — the distilled local model is a real, free, measurably-better-than-base
  option, and running it costs nothing per query.
- **For anything where quality actually matters** (the work you'd pay for Claude
  Code to get right) — cloud remains meaningfully better, confirmed by an
  independent blind judge, not close.
- **The one clear, actionable win:** if you're already committed to running
  locally for cost reasons, **distilling from a handful of cloud-quality
  examples is worth the 20-minute training cost** — it's free money on top of
  "running local," even though it doesn't make local competitive with cloud.

## Caveats

- **Small sample.** 6 eval questions, 20 training examples — the 6.83 vs 6.17
  vs 6.00 gap is a real, consistent signal (C won 5/6) but not statistically
  large; don't over-generalize the exact magnitude.
- **A 400-token generation cap** on all local conditions may have truncated
  some longer answers (the judge's notes mention this on a few responses) — this
  affected A/B/C roughly equally, so it likely doesn't bias the *relative*
  comparison between them, but does make all three look less complete than D's
  more naturally-bounded Sonnet answers.
- **One judge, one model family (Opus).** A second independent judge (a
  different model) would strengthen confidence further; not done here.
- **Distillation source matters.** This LoRA was trained on Sonnet's answers
  specifically — the quality ceiling is bounded by the teacher. A stronger
  teacher (or more examples) would likely widen C's lead over A/B further,
  though probably not close the D gap without much more data.

Receipts: `eval/receipts/soul_distill_eval.json`, `/tmp/sonnet_train.json` +
`/tmp/sonnet_eval.json` (gold answers, not committed — regenerable),
`data/soul_distill_prompts.json`, `data/soul_distill/`, adapter at
`dist/soul-distill-lora/`. Scripts: `scripts/57_soul_distill_build.py`,
`scripts/58_soul_distill_eval.py`.
