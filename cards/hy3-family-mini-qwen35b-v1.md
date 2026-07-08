---
license: apache-2.0
base_model: Qwen/Qwen3.6-35B-A3B
library_name: mlx
pipeline_tag: text-generation
language: en
tags:
  - mlx
  - apple-silicon
  - mixture-of-experts
  - agent
  - verifier-first
  - lora-fused
  - qwen3
---

# Hy3-Family Mini — Qwen35B v1

The **fast daily-agent tier** of the Hy3-Demolition family: `mlx-community/Qwen3.6-35B-A3B-4bit` fine-tuned with our **verifier-filtered LoRA** (the same repair / tool-discipline / strict-JSON / soul data that trained the Hy3 models), fused to a standalone **18 GB** MLX directory.

It is a **different base model** from Hy3 (Qwen3.6-35B-A3B, not the 295B Tencent Hy3) — a *sibling*, not a small Hy3 — carrying the same soul/verifier DNA on a right-sized base. Built on the **clean public Qwen base**.

## Why this exists

Streaming the real 295B Hy3 fits any Mac but runs at 0.7–3.85 tok/s — fine for one-off answers, too slow for an agent loop. This sibling is the **fast** answer: **100–367 tok/s**, fits a **32 GB or 64 GB** Mac, so it's a genuine interactive daily driver *and* daily agent (the agent's outputs are validated by deterministic verifiers anyway, so throughput + guardrails beat raw model size).

## ✅ Verified in LM Studio

Confirmed working in **LM Studio** (its bundled MLX engine = mlx-llm on mlx-lm 0.31.3): loads in ~9 s (18.2 GiB), generates correct output, clean stop. Just point LM Studio at this repo (arch `qwen3_5_moe` is natively supported).

Note: Qwen3.6 defaults to *thinking mode* — LM Studio routes the chain-of-thought to its reasoning panel (`reasoning_content`) and the answer to `content`. Toggle thinking off in LM Studio for direct answers, or give enough token budget for it to finish reasoning first.

## Usage

```bash
mlx_lm.chat --model <this-directory>
mlx_lm.server --model <this-directory> --port 8080   # OpenAI /v1
```
**Qwen3.6 quirk:** to turn OFF thinking, pass `enable_thinking=False` as a DIRECT kwarg to `apply_chat_template` (the nested form and `/no_think` are ignored).

## Verification (measured, receipts in the source repo)

- Heal: verifier-filtered LoRA, 8 layers, 600 iters, val loss 0.639.
- Stress test (10 hard prompts, **executed**): **8/10** — passes LRU cache, O(n) refactor, cross-field JSON, multi-constraint following, the bat-and-ball trap, destructive-request refusal, SQL-injection fix, thread-race fix; misses median-of-two-sorted (infinite loop) and a probability count. (`eval/receipts/sibling_qwen35b_stress_grade.json`)
- Smoke: correct code, clean stop.

## Limitations

- Different base than Hy3; not bit-comparable. English/agent focus. Cannot execute tools itself — pair with a real verifier harness (that's the design).
- MTP speculative-decoding variant (for MTPLX) is forged separately.

Source, recipe, receipts: https://github.com/PhilipJohnBasile/hy3-demolition-mlx
