# Day-To-Day Agent Target

The target is speed plus brains with the fewest runtime moving parts:

```text
fused Hy3 MLX model -> mlx_lm.server -> OpenAI-compatible clients
```

`agent-toolkit`, `agent-brain-blueprint`, `glm52-demolition`, and
`tinygpt-souls` are used before release to build verified data and heal the
model. They should not be required for normal daily use. (All but
`glm52-demolition` are private internal repos; not published.)

The release shape is:

```text
build-time repos -> verified SFT/LoRA heal -> fused MLX safetensors -> mlx_lm
```

No `agent-toolkit` import, soul verifier import, or demolition script should be
required to chat with the final fused model.

## Default Daily Path

1. Build or download the fused model (full lite-v1 recipe in RESTORE.md —
   train against the `-train` view, 8 layers, fuse with script 18):

```bash
. scripts/00_env.sh
./scripts/11_prepare_lite_sft.py
./scripts/15_import_glm52_datasets.py
./scripts/16_normalize_lite_sft_lengths.py --write
./scripts/17_prepare_hy3_train_view.py
./scripts/07_heal_lora_hy3_mlx.py \
  --model models/hy3-mlx-base-ar-train \
  --data data/hy3_lite_sft_combined \
  --adapter-path dist/adapters-hy3-lite-v1 \
  --iters 200 --num-layers 8 --max-seq-length 2048 \
  --train
./scripts/18_fuse_lora_streamed.py \
  --model models/hy3-mlx-base-ar \
  --adapter-path dist/adapters-hy3-lite-v1 \
  --save-path "$HY3_LITE_FUSED"
```

2. Serve for daily agent use:

```bash
./scripts/08_serve_mlx.py \
  --model "$HY3_LITE_FUSED" \
  --port 8080 \
  --max-tokens 4096 \
  --prefill-step-size 4096 \
  --prompt-cache-size 8 \
  --decode-concurrency 1 \
  --prompt-concurrency 1
```

3. Point clients at:

```text
http://127.0.0.1:8080/v1
```

Use the model id reported by `/v1/models`, or use the local model path in
clients that allow it.

## Direct Interactive Use

```bash
mlx_lm.chat --model "$HY3_LITE_FUSED"
```

For one-off smoke tests:

```bash
mlx_lm.generate \
  --model "$HY3_LITE_FUSED" \
  --prompt "Return strict JSON with keys ok and reason." \
  --max-tokens 128 \
  --temp 0
```

## Daily Settings

- Coding and agent work: temperature 0 to 0.2, top-p 1.0.
- Hard reasoning: set Hy3 chat-template reasoning effort in the request when supported.
- Long files: keep prompts structured, minimize repeated context, and keep the prompt cache enabled.
- Extended runs: leave other memory-heavy apps closed before loading the 295B MoE checkpoint.

## 128 GB Big-And-Fast Policy

This machine has enough unified memory to try the big MLX quantized Hy3 artifact
first. Do not prune the Lite model just to make it smaller.

Order of operations:

1. Load the existing mixed-quant Hy3 MLX checkpoint.
2. Measure one-token and short-generation latency.
3. Run `mlx_lm.server` with prompt cache and conservative concurrency.
4. Raise prefill step size and cache limits only if memory pressure stays sane.
5. Distill verifier-first behavior with LoRA and fuse it.
6. Start REAP pruning only after the big fused Lite model has receipts.

Initial daily server profile:

```bash
./scripts/08_serve_mlx.py \
  --model "$HY3_LITE_FUSED" \
  --port 8080 \
  --max-tokens 4096 \
  --prefill-step-size 4096 \
  --prompt-cache-size 8 \
  --decode-concurrency 1 \
  --prompt-concurrency 1
```

Faster settings are earned by receipts and memory observations, not assumed.
