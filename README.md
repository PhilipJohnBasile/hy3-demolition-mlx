---
license: apache-2.0
base_model: tencent/Hy3
library_name: mlx
pipeline_tag: text-generation
tags:
  - mlx
  - apple-silicon
  - hy3
  - mixture-of-experts
  - agent
  - verifier-first
  - research
---

# Hy3-Demolition-MLX

Apple-Silicon-first research pipeline for turning Tencent Hy3 into a local
verifier-trained agent brain while keeping the artifact MLX-native.

The end state is intentionally simple:

```bash
mlx_lm.chat --model dist/hy3-demolition-mlx-lite-fused
mlx_lm.server --model dist/hy3-demolition-mlx-lite-fused --port 8080
```

Everything else in this repo exists to put useful agent behavior into the model
before that final run command.

## Target

Hy3-Demolition-MLX is an Apple-Silicon-first research pipeline that turns
Tencent Hy3 into a local verifier-first agent model. It keeps the full path
MLX-native: streamed expert saliency, conservative REAP pruning, mixed
precision MLX quantization, LoRA healing on verified agent/code data, and
OpenAI-compatible local serving through `mlx_lm.server`.

## Rules

1. Runtime: MLX only
2. Weight format: MLX safetensors only
3. Calibration: `mlx.core` / `mlx.nn` only
4. Pruning: stream one MLX layer or shard at a time
5. Quantization: `mlx.quantize` / `mlx.dequantize` only
6. Serving: `mlx_lm.generate`, `mlx_lm.chat`, and `mlx_lm.server`
7. Agent layer: build-time verifier/curriculum, not a required runtime wrapper
8. No GGUF
9. No llama.cpp
10. No CUDA dependency

## What Goes Into The Model

The final goal is not "Hy3 plus a wrapper." It is a fused MLX model artifact.

Build-time sources:

- `agent-toolkit`: verifier mesh used to keep only passing code, tool-call, JSON,
  repair, and domain outputs.
- `agent-brain-blueprint`: behavior curriculum for local LLM engineering,
  verifier-first operation, agent security, tool-use discipline, and repair loops.
- `glm52-demolition`: proven demolition pattern, eval discipline, REAP/LoRA
  framing, and verified-data shape.
- `tinygpt-souls`: soul tags, canons, and per-soul verifier-gated examples.
- `glm52-demolition-data`: reusable verified heal, soul, agentic, and REAP
  calibration data from the GLM demolition run.
- `glm52-verified-fixes`: private execution-verified bug-fix examples for repair
  LoRA heal.

The source manifest is tracked at `data/build_time_sources.json`. It is the
runtime boundary: every listed repo contributes data, gates, curriculum, or
recipes before release, but the final target remains one fused MLX model
directory with no required wrapper.

The GLM dataset import plan is tracked at `data/glm52_dataset_import_plan.json`.
It should be sampled and normalized; it should not be dumped wholesale into Hy3.

What can be distilled into weights:

- tool-call format discipline
- coding and repair habits
- verifier-first self-checking behavior
- brain-blueprint operating stance
- soul/facet routing prompts such as `<|soul:security|>` and `<|soul:music|>`
- concise failure reporting and retry behavior

What cannot honestly be baked into weights:

- real compiler execution
- filesystem or shell authority
- exact test results for unseen code
- external retrieval or private memory

Those remain build-time gates and optional eval tools. The shipped model should
still run without importing `agent-toolkit`.

## Phases

### `hy3-demolition-mlx-lite`

Base:

- existing Hy3 MLX quantized checkpoint

Demolition:

- no pruning

Model-first agent work:

- generate verifier-filtered SFT data from `agent-toolkit`
- distill `agent-brain-blueprint` and `tinygpt-souls` behavior into LoRA
- fuse the LoRA into a standalone MLX model directory
- validate with direct `mlx_lm.generate`, `mlx_lm.chat`, and `mlx_lm.server`

### `hy3-demolition-mlx-reap`

Base:

- Hy3 MLX checkpoint

Demolition:

- Hy3-specific streamed REAP saliency
- conservative 25 percent expert prune first
- 40 percent prune only after receipts prove agent reliability holds
- mixed precision MLX requantization
- LoRA heal on verified agent/code data
- fuse before release

## Metal On Apple Silicon

There is no CUDA switch in this project. MLX uses Metal automatically when it is
installed in an arm64 Python environment on Apple Silicon.

Verify it before loading Hy3:

```bash
.venv/bin/python -c "import mlx.core as mx; print('metal:', mx.metal.is_available()); print(mx.device_info())"
```

Expected on this Mac:

```text
metal: True
```

If that prints `False`, fix the environment before serving the model:

- use the native arm64 Homebrew/Python, not Rosetta
- install `mlx` and `mlx-lm` in the active venv
- run model-loading commands from a normal terminal session with GPU access
- keep the MLX safetensors checkpoint local on fast storage

Inside restricted automation sandboxes, Metal can be hidden even when the host
supports it. The actual host probe for this workspace reports Metal available on
Apple M5 Max with 128 GB unified memory.

## Quick Start

```bash
. scripts/00_env.sh
./scripts/01_download_mlx_base.sh
./scripts/02_inspect_hy3_mlx.py --model "$HY3_MODEL_DIR"
./scripts/13_prepare_hy3_ar_view.py --source "$HY3_MODEL_DIR" --out models/hy3-mlx-base-ar
./scripts/13_smoke_generate_mlx_ar.py --model models/hy3-mlx-base-ar --max-tokens 16
./scripts/08_serve_mlx.py --model models/hy3-mlx-base-ar --port 8080 --prompt-cache-size 8
./scripts/03_baseline_agent_eval.py --base-url http://127.0.0.1:8080/v1
```

Current verified Lite baseline:

- `models/hy3-mlx-base`: 105 GB MLX safetensors, `hy_v3`, 80 layers
- `models/hy3-mlx-base-ar`: symlinked AR-only view with `num_nextn_predict_layers=0`
- MTP sidecar omitted for the day-to-day path
- no-wire MLX generation avoids the Metal watchdog seen with the stock giant-model pin
- one-token smoke: `ready`, peak memory 112.329 GB
- sixteen-token smoke: coherent sentence, peak memory 112.329 GB, 16 tokens in 49.5 s end-to-end on a cold load (see receipt; steady-state decode measured separately at ~7.4 tok/s warm in hy3_mtp_smoke.json)

Receipts:

- `eval/receipts/hy3_base_inspect.json`
- `eval/receipts/hy3_ar_smoke.json`
- `eval/receipts/hy3_ar_smoke_16.json`
- `eval/receipts/hy3_server_smoke.json`

Promoted lite-v1 (2026-07-07): trained 200 iters on the length-normalized
combined pack (val loss 1.154 → 0.722), fused, and verified standalone.
Agent eval passed 5/5 with the adapter and 5/5 fused through
`/v1/chat/completions`; peak memory 112.3 GB; ~7.4 tok/s warm decode (hy3_mtp_smoke.json).

- `eval/receipts/hy3_lite_sft_length_normalization.json`
- `eval/receipts/hy3_lite_v1_train.json`
- `eval/receipts/hy3_lite_v1_adapter_smoke.json`
- `eval/receipts/hy3_lite_v1_adapter_eval.jsonl`
- `eval/receipts/hy3_lite_v1_fuse.json`
- `eval/receipts/hy3_lite_v1_fused_smoke.json`
- `eval/receipts/hy3_lite_v1_fused_eval.jsonl`
- `eval/receipts/hy3_lite_v1_fused_server_smoke.json`

## Eval Tiers

Two tiers, one runner (`scripts/09_eval_agent_toolkit.py`):

- **Fast tier** (~20 min warm): the 15 short-form cases — coding, tool-call
  payloads, repair, strict JSON. For iterating on adapters, quant policies,
  and prune candidates.

  ```bash
  ./scripts/09_eval_agent_toolkit.py \
    --cases eval/coding/prompts.jsonl eval/tool_calls/prompts.jsonl \
            eval/agent_repair/prompts.jsonl eval/json_schema/prompts.jsonl \
    --out eval/receipts/fast_tier.jsonl
  ```

- **Full suite** (~4 h): all 30 cases including planning and the 11 protected
  souls. Required for promotion gates; compared against the committed baseline
  with `scripts/20_compare_receipts.py` (exit 0 = promote).

## Soul-Preserving REAP

Hy3 pruning must not optimize only for average expert saliency. Rare souls can
be low-frequency and still important. Before any expert cut, run calibration
with `eval/souls/protected_prompts.jsonl`; prune planning then reserves each
protected soul's top routed experts per layer before choosing aggregate drops.

The default protected souls are coding, math, science, security, design,
fullstack, gamedev, legacy, music, art, and perfumery. `scripts/05_apply_reap_prune_hy3_mlx.py`
refuses to build a protected prune plan unless those soul saliency buckets are
present, unless an explicit override is passed.

Train and fuse the Lite behavior adapter (lite-v1 recipe):

```bash
./scripts/11_prepare_lite_sft.py
./scripts/15_import_glm52_datasets.py
./scripts/16_normalize_lite_sft_lengths.py --write
./scripts/17_prepare_hy3_train_view.py
./scripts/07_heal_lora_hy3_mlx.py \
  --model models/hy3-mlx-base-ar-train \
  --data data/hy3_lite_sft_combined \
  --adapter-path dist/adapters-hy3-lite-v1 \
  --iters 200 \
  --num-layers 8 \
  --max-seq-length 2048 \
  --train
./scripts/18_fuse_lora_streamed.py \
  --model models/hy3-mlx-base-ar \
  --adapter-path dist/adapters-hy3-lite-v1 \
  --save-path dist/hy3-demolition-mlx-lite-v1-fused
```

Three hard-won rules baked into that recipe:

- Train against `models/hy3-mlx-base-ar-train` (script 17), never the plain AR
  view: the stock Hy3 chat template only appends EOS to the final assistant
  turn when `is_training` is set, and training without it destroys stop
  behavior (token-0 `!` spam after answers).
- `--num-layers 16` does not fit the 128 GB envelope for LoRA training on this
  checkpoint (Metal OOM); 8 layers trains fine.
- Fuse with script 18 (lazy, shard-streamed), not `mlx_lm fuse`, which loads
  all 105 GB eagerly and dies or thrashes swap.

Run the fused model directly:

```bash
mlx_lm.generate --model dist/hy3-demolition-mlx-lite-v1-fused --prompt "Write a tiny Python fizzbuzz." --max-tokens 128
mlx_lm.chat --model dist/hy3-demolition-mlx-lite-v1-fused
mlx_lm.server --model dist/hy3-demolition-mlx-lite-v1-fused --port 8080
```

## Repo Map

```text
scripts/   numbered build, serve, eval, prune, quant, heal steps
src/       MLX weight-store, REAP, serving, prompt, receipt helpers
data/      seed and imported build-time SFT/calibration packs
eval/      tiny verifier-first smoke suites and receipts
dist/      generated fused artifacts, adapters, and plans
models/    downloaded MLX base checkpoints
docs/      upstream contribution patches and design docs
```

Governance docs (read these before doing anything substantial):

- `BACKLOG.md` — numbered status board + execution order and estimates
- `DECISIONS.md` — pre-committed judgment for every REAP/promotion call
- `BUILD_NOTES.md` — append-only incident log (what went right and wrong)
- `RESTORE.md` — rebuild everything from source, including local patches

## External Model Facts

Tencent's Hy3 model card describes Hy3 as a 295B-parameter MoE with 21B active
parameters, 80 layers, 192 experts with top-8 routing, 256K context, and BF16
official precision. It also recommends large-memory multi-GPU serving for the
official deployment path. This repo targets the local Apple Silicon path instead:
MLX quantized artifacts, then conservative MLX-native demolition only after the
Lite model proves reliable.
