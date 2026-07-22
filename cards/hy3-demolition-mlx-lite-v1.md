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
---

# Hy3-Demolition-MLX lite-v1

A standalone, fused MLX model directory built from Tencent Hy3 (295B MoE, 21B
active, 192 experts / top-8, 80 layers) for local Apple Silicon use. A LoRA
adapter trained on verifier-filtered agent data (coding, repair, tool-call
discipline, strict JSON, security, and domain facets) is fused into
the quantized weights. No experts are pruned in the lite variant.

This is a **research artifact**, shared as-is with receipts: no support or
fitness claims, Apache-2.0 like its base model. Every number in this card is
measured, not estimated — the receipts live in the source repo.

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
(measured peak 112.3 GB on an M5 Max 128 GB).

```bash
mlx_lm.generate --model <this-directory> --prompt "Write a tiny Python fizzbuzz." --max-tokens 512
mlx_lm.chat --model <this-directory>
mlx_lm.server --model <this-directory> --port 8080   # OpenAI-compatible /v1
```

No adapter path, agent framework, or custom runtime wrapper is required —
the directory is the whole artifact. For clean, direct answers, serve with
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

**Serving multiple concurrent requests:** `mlx_lm.server` (above) does batch
requests, but was measured unreliable at 8-way concurrency on the reap25
tier (dropped ~25% of connections under load) — solid at ≤4 concurrent
(2.72× aggregate, 4/4 completed). For reliable high-concurrency serving,
[oMLX](https://omlx.ai/) is a viable alternative: loads with one dependency
swap (install this project's pinned `mlx-lm` fork into its venv instead of
its default mainline pin — `RESTORE.md` has the exact command), measured
3.49× aggregate at 8/8 concurrent with no drops on reap25, plus active
memory-ceiling enforcement and a restart-surviving KV cache. Not a blanket
"faster" claim — single-stream speed was comparable either way in that
test. These numbers were measured on the smaller reap25 tier (87 GB); this
card's larger 112 GB checkpoint should behave the same qualitatively but
hasn't been separately benchmarked.

## Source and recipe

- Base: `ox-ox/Hy3-295B-Instruct-w2q3exp-AProjQ8-SExpQ8-OutQ8-MTP-mlx`
  (mixed-quant MLX checkpoint of Tencent Hy3), used as an AR-only view
  (MTP sidecar omitted, `num_nextn_predict_layers=0`).
- LoRA: rank 8 on 8 layers (211M trainable params, 0.072%), 200 iterations,
  batch 1, lr 1e-5, adamw, gradient checkpointing, masked prompts,
  max-seq 2048, val loss 1.154 → 0.722.
- Training data: 305/16/16 verifier-gated examples (repair, coding, agentic,
  security, music, perfumery facets), length-normalized so nothing truncates.
- Fused with a streamed lazy fuse (one shard at a time), keeping the stock
  Hy3 chat template.

Build scripts, receipts, and the full recipe:
https://github.com/PhilipJohnBasile/hy3-demolition-mlx (tag `lite-v1`).

## Run on a smaller Mac — SSD expert-streaming (16–128 GB)

Fully resident this model needs ~112 GB. But with the **SSD expert-streaming
pager** (`src/hy3_streaming.py` in the source repo), the same weights run on far
smaller Macs — only the ~9.7 GB of non-expert weights + a small LRU of hot
experts stay in RAM, and the routed experts are paged from disk per token,
**bit-identical (zero quality loss)**. Because streaming only touches the 8
routed experts per token regardless of the total (192 here), the resident
profile matches the pruned reap25 tiers (16 GB ≈ 12 GB peak, 64 GB ≈ 23 GB peak;
only the on-disk size differs). See `scripts/41_streaming_load.py`. Experimental
serving path; the fully-resident `mlx_lm` path is the default.

## Verification receipts

- Fresh full baseline (suite+hard+brutal) 46/46 (2026-07-08); receipts in the source repo.

All measured on the fused artifact, 2026-07-07, receipts committed under
`eval/receipts/` in the source repo:

- Direct `mlx_lm.generate` smoke: correct output, clean EOS stop.
- Agent eval 5/5 through `mlx_lm.server` `/v1/chat/completions`
  (coding ×2, tool-call JSON, repair, strict JSON schema), outputs verified
  by the agent-toolkit verifier mesh (private internal tooling; not published).
- Strict-JSON server smoke: parseable JSON, `finish_reason: stop`.
- Peak inference memory 112.3 GB; warm decode ~7.4 tok/s (measured, `hy3_mtp_smoke.json`). First-token latency is high on cold load.

## Limitations

- The model learned self-checking and repair *habits*; it cannot actually
  execute compilers, tests, shells, or tools. Verification of its outputs
  still needs a real harness.
- Quantized MoE base: expect the usual quantization artifacts.
- Trained/evaluated primarily in English on agent/code tasks.
- ~7.4 tok/s warm decode on M5 Max (measured) — a usable local daily
  driver with prompt caching, not a high-throughput server. Cold load
  pages ~104 GB from disk, so the first response lags.

## Runtime support

Supported: `mlx_lm.generate`, `mlx_lm.chat`, `mlx_lm.server`.
Not supported (today): GGUF, llama.cpp, vLLM, SGLang, CUDA serving.
- **LM Studio / Ollama:** pending — needs the `hy_v3` architecture in *mainline* mlx-lm ([ml-explore/mlx-lm#1211](https://github.com/ml-explore/mlx-lm/pull/1211)). Today these load only on the pinned fork (`eauchs/mlx-lm@hy_v3-mtp`); once #1211 merges, the AR models should work in LM Studio.
