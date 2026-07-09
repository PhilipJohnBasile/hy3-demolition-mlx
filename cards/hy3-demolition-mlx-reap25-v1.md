---
license: apache-2.0
base_model: tencent/Hy3
library_name: mlx
pipeline_tag: text-generation
language: en
tags:
  - mlx
  - apple-silicon
  - hy3
  - mixture-of-experts
  - agent
  - verifier-first
  - lora-fused
  - reap-pruned
---

# Hy3-Demolition-MLX reap25-v1

A standalone, fused MLX model directory built from Tencent Hy3 (295B MoE, 21B
active, 192 experts / top-8, 80 layers), **rare-expert-preserving REAP-pruned to 144
experts per layer (25%)** and then healed with a LoRA on verifier-filtered
agent data. It is the same recipe as `lite-v1` plus the prune — a smaller,
lighter daily driver (**223B params, ~87 GB peak** vs lite-v1's 295B / 112 GB)
that measures functionally indistinguishable from the unpruned model.

This is a **research artifact**, shared as-is with receipts: no support or
fitness claims, Apache-2.0 like its base model. Every number here is measured,
not estimated — receipts live in the source repo.

## Why reap25 (and why not deeper)

REAP (arXiv:2510.13999) ranks experts by the mean over routed tokens of
`gate_value × ‖expert_output‖₂` — a criterion that decouples frequency from
impact, so rare high-impact experts are protected. The full family was built and
measured on an M5 Max 128 GB:

| tier | eval | peak mem | healed val loss | verdict |
|---|---|---|---|---|
| lite-v1 (no prune) | 46/46 | 112 GB | 0.722 | reference |
| **reap25 (this)** | **45/46** | **86.7 GB** | **0.979** | **the keeper** |
| reap40 (40%) | 45/46 | 71 GB | 1.078 | rejected (real code-case crash) |

reap25's single eval miss was a *truncation* of an otherwise-correct answer,
not a wrong answer; it held every hard and brutal code/tool/JSON case (8/8,
8/8). reap40 scored the same count but failed a brutal code case with a runtime
crash — so **25% is the pruning knee**; deeper is not shipped.

## Usage

> ⚠️ **Runtime prerequisite — read first.** This is a `hy_v3` (Tencent Hy3)
> model. `hy_v3` is **not in mainline `mlx-lm` yet** ([ml-explore/mlx-lm#1211](https://github.com/ml-explore/mlx-lm/pull/1211)),
> so **stock `mlx-lm`, LM Studio, and Ollama cannot load it** (you'll get
> `ValueError: Model type hy_v3 not supported`). Run it with the **pinned fork**:
> ```bash
> pip install "mlx-lm @ git+https://github.com/eauchs/mlx-lm@hy_v3-mtp"
> ```
> (See RESTORE.md in the source repo for the exact commit + the fp32-router patch.)
> Once #1211 merges, mainline `mlx-lm` and LM Studio will load it with no changes.
> — For a model that runs on **stock mlx-lm / LM Studio today**, use the fast
> sibling [`hy3-family-mini-qwen35b-v1`](https://huggingface.co/philipjohnbasile/hy3-family-mini-qwen35b-v1).

MLX only. Requires an Apple Silicon Mac with enough unified memory
(measured peak 86.7 GB — leaves ~41 GB free on a 128 GB machine).

```bash
mlx_lm.generate --model <this-directory> --prompt "Write a tiny Python fizzbuzz." --max-tokens 512
mlx_lm.chat --model <this-directory>
mlx_lm.server --model <this-directory> --port 8080   # OpenAI-compatible /v1
```

No adapter path, agent framework, or custom runtime wrapper is required — the
directory is the whole artifact. For clean, direct answers, serve with
`--chat-template-args '{"reasoning_effort":"no_think"}'`; use `high` for hard
agent tasks where step-by-step reasoning helps (the model reasons inside
`<think:opensource>…</think:opensource>` tags). Suggested settings for
coding/agent work: temperature 0–0.2, top-p 1.0.

⚠️ **`max_tokens` too low fails *silently*, not loudly.** With reasoning left
on (the default), a small token budget can be entirely consumed by the
`<think:opensource>` trace, leaving **zero output with no error** — this
looks like a clean empty result, not a truncation. Confirmed on the sibling
model in a real production pipeline (a `max_tokens: 1600` cap returned zero
results for every input before the cause was found; `7000–9000` fixed it).
If output looks suspiciously empty, raise `max_tokens` or set
`reasoning_effort: no_think` before assuming the task legitimately had
nothing to produce.

⚠️ **For strict-JSON/structured output, pin `repeat_penalty: 1.0` explicitly
in the request — don't trust a server's UI/global default.** A nonzero
repeat penalty punishes exactly the repeated structural tokens JSON needs
(`"name":`, `"type":`, array delimiters), and can silently corrupt output on
batch/array-heavy extraction tasks. Confirmed in production: a serving UI
had `repeat_penalty: 1.1` active globally, invisible to the request itself,
on a strict-JSON extraction workload. A pipeline's full sampling contract
(temperature, top-k, repeat-penalty) belongs in every request body, never
assumed from server state — server defaults and UI settings can change
between calls or restarts without you noticing.

## Run on a smaller Mac — SSD expert-streaming (16–128 GB)

Fully resident this model needs ~87 GB. But with the **SSD expert-streaming
pager** (`src/hy3_streaming.py` + `scripts/41_streaming_load.py` in the source
repo), the *same weights* run on far smaller Macs: only the ~9.7 GB of
non-expert weights plus a small LRU of hot experts stay in RAM, and the routed
experts are paged from disk per token — **bit-identical, zero quality loss**
(the experts are the exact same weights, just read on demand).

| Mac | LRU cache | peak resident | decode |
|---|---|---|---|
| 64 GB | 24 experts/proj | 23.4 GB | 3.85 tok/s |
| 32 GB | 6 experts/proj | 13.9 GB | 0.81 tok/s |
| 16 GB | 2 experts/proj | 11.7 GB | 0.74 tok/s |

Measured on an M5 Max; one `STREAM_CACHE` dial trades memory for speed. This is
an **experimental serving path** (a research prototype, not a packaged engine);
the fully-resident `mlx_lm` path above is the default.

## Source and recipe

- Base: `ox-ox/Hy3-295B-Instruct-w2q3exp-AProjQ8-SExpQ8-OutQ8-MTP-mlx`
  (mixed-quant MLX checkpoint of Tencent Hy3), AR-only view
  (`num_nextn_predict_layers=0`).
- REAP calibration: streamed, true-criterion, facet-bucketed over a
  domain-matched pack; all 11 domain facets present in the saliency.
- Prune: 192 → 144 experts/layer (25%), rare-expert-preserving plan verified by the
  analyzer (ACCEPT — 87% average saliency mass kept, no layer below 70%, all
  facets protected in every layer). Pruned tensors keep their original
  quantization (no dequant→requant cycle).
- Heal LoRA: rank 8 on 8 layers, 200 iterations, batch 1, lr 1e-5, adamw,
  max-seq 2048, trained against the `is_training` template view (EOS-safe),
  val loss 1.566 → 0.979.
- Fused with a streamed lazy fuse (one shard at a time), stock Hy3 chat
  template.

Build scripts, receipts, and the full recipe:
https://github.com/PhilipJohnBasile/hy3-demolition-mlx

## Verification receipts

All measured on the fused artifact, 2026-07-08, receipts committed under
`eval/receipts/` in the source repo:

- Full eval (suite + hard + brutal) **45/46**; hard 8/8, brutal 8/8; the one
  miss is a token-cap truncation of a correct domain answer, not a wrong answer
  (`hy3_reap25_eval.jsonl`, `hy3_reap25_vs_lite_compare.json`).
- Manual side-by-side vs lite-v1 on real code / planning / repair / domain
  prompts (production `no_think` mode): clean, correct, direct, functionally
  indistinguishable — one marginal edge each way
  (`reap25_vs_lite_side_by_side.md`).
- Prune-plan analyzer verdict: ACCEPT (`hy3_reap25_plan_report.json`).
- Pruned + fused smokes: correct output, clean EOS stop
  (`hy3_reap25_pruned_smoke.json`, `hy3_reap25_fused_smoke.json`).
- Peak inference memory 86.7 GB.

## Limitations

- The model learned self-checking and repair *habits*; it cannot execute
  compilers, tests, shells, or tools. Verifying its outputs still needs a real
  harness.
- Quantized, pruned MoE base: expect quantization artifacts, and a small
  capability cost from the 25% prune (visible as +0.26 healed val loss over
  lite-v1, though not as felt output-quality loss in the manual pass).
- Trained/evaluated primarily in English on agent/code tasks.
- Decode speed is unchanged from lite-v1 (top-8 × 21B active is the same); the
  win is ~25 GB of memory headroom, not throughput.
- **Tool-calling tag format regression (found 2026-07-08, isolated against
  lite-v1):** when emitting a native `<tool_call:opensource>` block, this
  model unreliably omits the `<tool_sep:opensource>` tag between the function
  name and its arguments — confirmed across 2 independent tool schemas/prompts.
  `arg_key`/`arg_value` pairs still parse correctly, but strict-format parsers
  expecting `<tool_call>NAME<tool_sep>...` will mis-extract the function name.
  lite-v1 (same LoRA heal, no prune) does not show this — it appears specific
  to the 25% prune, not the heal, and wasn't caught by our eval suite (no test
  case exercised this exact tag boundary). Workaround: extract `name` via a
  looser regex, or fall back to `arg_key`/`arg_value` pairs plus a
  looked-up/known tool name until a fix ships.

## Runtime support

Supported: `mlx_lm.generate`, `mlx_lm.chat`, `mlx_lm.server`.
Not supported (today): GGUF, llama.cpp, vLLM, SGLang, CUDA serving.
- **LM Studio / Ollama:** pending — needs the `hy_v3` architecture in *mainline* mlx-lm ([ml-explore/mlx-lm#1211](https://github.com/ml-explore/mlx-lm/pull/1211)). Today these load only on the pinned fork (`eauchs/mlx-lm@hy_v3-mtp`); once #1211 merges, the AR models should work in LM Studio.
