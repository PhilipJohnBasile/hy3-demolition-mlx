# Hy3-Demolition-MLX Model Card

## Model Summary

Hy3-Demolition-MLX is a local MLX artifact built from Tencent Hy3 for
Apple Silicon. The intended final artifact is a fused MLX safetensors model
directory that can be loaded directly by `mlx_lm` without a custom runtime
wrapper.

## Base Model

- Base family: Tencent Hy3
- Architecture: Mixture of Experts
- Published scale: 295B total parameters, 21B active parameters
- Published routing: 192 experts, top-8 activated
- Published context: 256K
- Published precision: BF16
- Local Lite base: `ox-ox/Hy3-295B-Instruct-w2q3exp-AProjQ8-SExpQ8-OutQ8-MTP-mlx`

## Intended Variants

### Lite

- Starts from an existing Hy3 MLX quantized checkpoint.
- Does not prune experts.
- Distills local agent behavior, tool-call format discipline, repair style,
  and soul/facet control into a LoRA.
- Fuses the LoRA into a standalone MLX model directory.

### REAP

- Streams Hy3 MLX safetensors.
- Scores expert saliency with real routed activations.
- Separately records protected soul/facet routing so rare souls are not pruned
  away by average-case REAP.
- Applies conservative 25 percent expert pruning before any larger cut.
- Requantizes with MLX primitives.
- Heals with verified code/agent data.
- Fuses adapters before release.

## Build-Time Teachers And Gates

- `agent-toolkit`: verifier mesh for code, SQL, design, math, tool-call, and
  repair data filtering.
- `agent-brain-blueprint`: operating stance, verification behavior, agent
  security, local LLM engineering, and tool-use curriculum.
- `glm52-demolition`: demolition pattern and measurement discipline.
- `tinygpt-souls`: soul tags, canons, and per-soul verifier patterns.

These are not required at runtime for the fused model.

The build-time source manifest is `data/build_time_sources.json`. The release
goal is still a standalone fused MLX model, not a model plus those repositories.

## Runtime

Supported:

- `mlx_lm.generate`
- `mlx_lm.chat`
- `mlx_lm.server`

Not supported:

- GGUF
- llama.cpp
- Ollama
- vLLM
- SGLang
- CUDA-first serving

## Released Artifacts

### lite-v1 (2026-07-07)

- Artifact: `dist/hy3-demolition-mlx-lite-v1-fused` (104 GB MLX safetensors,
  standalone, stock chat template).
- Recipe: length-normalized combined SFT pack (305/16/16, all ≤ 2048 tokens),
  LoRA rank 8 on 8 layers (211M params, 0.072%), 200 iters, batch 1, lr 1e-5,
  adamw, grad checkpoint, masked prompts, trained against the `is_training`
  template view; streamed lazy fuse.
- Val loss 1.154 → 0.722. Agent eval 5/5 with adapter and 5/5 fused via
  `mlx_lm.server` `/v1/chat/completions`. Peak inference memory 112.3 GB on an
  M5 Max 128 GB; ~1.4 tok/s cold generation.

## Evaluation Receipts

Receipts are JSONL files under `eval/receipts/`. Each record should include:

- prompt id
- model id or local model path
- backend (`mlx_lm`)
- generation text or tool-call payload
- verifier result
- pass/fail
- timestamp

The project policy is to report measured results only.

## Limitations

The model can learn the habit of self-checking and structured repair, but it
cannot internally execute compilers, shell commands, tests, or external tools.
Those remain build-time filters or optional runtime agent harnesses.

## Day-To-Day Agent Use

Serve:

```bash
mlx_lm.server --model <fused-mlx-model> --port 8080
```

Interactive:

```bash
mlx_lm.chat --model <fused-mlx-model>
```
