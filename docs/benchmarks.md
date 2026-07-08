# Benchmark report — Hy3 family, measured 2026-07-08 (M5 Max 128 GB)

All "ours" numbers are **measured on our own harness** (`scripts/45`–`51`),
zero-shot chat-prompted, `no_think` unless noted. Cloud/frontier numbers are
**published** (their protocols) — **not directly comparable**; see caveats.

## Our measured results

| Benchmark | sibling (healed) | native Qwen base | what it tests |
|---|---|---|---|
| HumanEval pass@1 | 92.7% | **94.5%** | code (saturated) |
| MBPP pass@1 | **82.9%** | 82.5% | code (saturated) |
| GSM8K | ~90% (partial)* | ~90%* | math (saturated) |
| AIME'24 (no_think) | **6.7%** | 0.0% | competition math (discriminating) |
| AIME'24 (thinking) | 3.3% | 3.3% | " with reasoning on |
| GPQA-diamond | **4.0%** | 3.0% | grad science (discriminating) |
| Souls (11 facets) | 10/11 | **11/11** | on-distribution (what we healed) |

*GSM8K runs were stopped early to prioritize other work; both tracked ~90%.

## The two findings that matter

1. **The heal is ≈ neutral.** Same ruler, base vs sibling: −1.8 HumanEval, +0.4
   MBPP, +6.7 AIME(nt), +1.0 GPQA, −1 soul. It neither meaningfully helped nor
   hurt raw quality — on academic *and* on-distribution (souls) tasks. See
   [souls-deep-dive.md](souls-deep-dive.md): the base Qwen3.6 already masters the
   soul facets, so a light LoRA had little room to add capability.
2. **Small models floor on hard reasoning, and thinking doesn't rescue them.**
   AIME ~0–7%, GPQA ~4% vs frontier ~90%+. Thinking mode changed AIME from 6.7%
   to 3.3% (noise) at 8× the tokens — a 3B-active model can't reason its way to
   capability it lacks.

## vs the field (published — different rulers)

| Model | HumanEval | GSM8K | AIME | SWE-bench |
|---|---|---|---|---|
| Opus 4.x / GPT-5 / Gemini 3 | ~95%+ | ~95%+ | ~90–100% | **74–82%** |
| Native Hy3 (our base's parent) | n/r | 95.4 | n/r | 74.4 |
| Kimi K2.5 | ~99% | ~88% | — | high |
| DeepSeek-V3 | ~79–83% | ~89% | — | — |
| Qwen2.5-72B | ~74% | ~88% | — | — |
| **our sibling** *(our ruler)* | 92.7% | ~90% | 6.7% | — |

### Caveats (read before comparing)

- **Different rulers.** Our harness is lenient zero-shot chat; published numbers
  use stricter EvalPlus-style protocols. Our 92.7% ≠ their 80% on the same scale.
- **HumanEval/GSM8K are saturated** (~90%+ for everything good) → they don't rank
  at the top. The frontier is decided on **SWE-bench / LiveCodeBench / AIME /
  GPQA**, where our small models trail badly and honestly.
- **We cannot run SWE-bench locally** (Docker + repo harness per instance) — that
  row is published-only.

## Honest one-liner

An **excellent small, local, private** model: near the ceiling on saturated
benchmarks, at the floor on the discriminating ones. Its value is *runs-on-your-
Mac*, not raw capability. The heal made it **local + consistently formatted**, not
smarter — the base Qwen3.6 was already strong.

Receipts: `eval/receipts/{humaneval,mbpp,gsm8k,aime,aime_think,gpqa,souls}_*.json`.
Harnesses: `scripts/45_humaneval.py`, `46_gsm8k.py`, `47_mbpp.py`, `49_aime.py`,
`50_gpqa.py`, `51_souls.py`, `43_shootout.py`.
