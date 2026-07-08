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
discipline, strict JSON, security, and protected soul facets) is fused into
the quantized weights. No experts are pruned in the lite variant.

This is a **research artifact**, shared as-is with receipts: no support or
fitness claims, Apache-2.0 like its base model. Every number in this card is
measured, not estimated — the receipts live in the source repo.

## Usage

MLX only. Requires an Apple Silicon Mac with enough unified memory
(measured peak 112.3 GB on an M5 Max 128 GB).

```bash
mlx_lm.generate --model <this-directory> --prompt "Write a tiny Python fizzbuzz." --max-tokens 128
mlx_lm.chat --model <this-directory>
mlx_lm.server --model <this-directory> --port 8080   # OpenAI-compatible /v1
```

No adapter path, agent framework, or custom runtime wrapper is required —
the directory is the whole artifact. Suggested settings for coding/agent
work: temperature 0–0.2, top-p 1.0.

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

## Verification receipts

- Fresh full baseline (suite+hard+brutal) 46/46 (2026-07-08); receipts in the source repo.

All measured on the fused artifact, 2026-07-07, receipts committed under
`eval/receipts/` in the source repo:

- Direct `mlx_lm.generate` smoke: correct output, clean EOS stop.
- Agent eval 5/5 through `mlx_lm.server` `/v1/chat/completions`
  (coding ×2, tool-call JSON, repair, strict JSON schema), outputs verified
  by the agent-toolkit verifier mesh.
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
