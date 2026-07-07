# Restore

This repo is designed so the local generated artifacts can be deleted and
rebuilt from source plus Hugging Face checkpoints.

## Environment

```bash
uv venv --python 3.12 --seed
. .venv/bin/activate
pip install "transformers>=5.7,<5.13" \
  "mlx-lm @ git+https://github.com/eauchs/mlx-lm.git@a7cc3054b1ff48c19950513b422a66cfed7baa60"
```

The mlx-lm pin is the exact commit of the `hy_v3-mtp` branch this project was
built and verified against (mlx 0.31.2, mlx-lm 0.31.3, transformers 5.12.1 —
see `eval/receipts/hy3_environment.json`). The branch moves; the EOS template
behavior, streamed fuse, and hy_v3/MTP support are all version-specific, so
restore from the commit hash, not the branch name.

After installing, re-apply the fp32 router patch (the reference Hy3
implementation computes the gate matmul in fp32; bf16 perturbs top-8 expert
selection — see DECISIONS.md D0). One-line change in
`mlx_lm/models/hy_v3.py` `MoEGate.__call__`:
`self.gate(x)` → `self.gate(x.astype(mx.float32))`
(upstream candidate: avlp12/mlx-lm@14f7837).

Then:

```bash
. scripts/00_env.sh
```

## Restore Lite Base

```bash
./scripts/01_download_mlx_base.sh
./scripts/02_inspect_hy3_mlx.py --model "$HY3_MODEL_DIR"
./scripts/13_prepare_hy3_ar_view.py --source "$HY3_MODEL_DIR" --out models/hy3-mlx-base-ar
./scripts/13_smoke_generate_mlx_ar.py --model models/hy3-mlx-base-ar --max-tokens 16
./scripts/08_serve_mlx.py --model models/hy3-mlx-base-ar --port 8080 --prompt-cache-size 8
```

The AR view is intentionally lightweight: it symlinks the base checkpoint,
omits `model-mtp.safetensors`, sets `num_nextn_predict_layers=0`, and writes a
tokenizer config with `fix_mistral_regex=true`.

## Rebuild Lite Fused Artifact

Prepare verified SFT data in `data/hy3_lite_sft_combined/{train,valid,test}.jsonl`, then:

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

Do not train against the plain AR view (its chat template omits EOS on the
final assistant turn unless `is_training` is set, which breaks stop behavior),
do not raise `--num-layers` past 8 on a 128 GB machine, and do not use
`mlx_lm fuse` (eager load of the full 105 GB checkpoint; script 18 streams).

Run:

```bash
mlx_lm.chat --model dist/hy3-demolition-mlx-lite-v1-fused
mlx_lm.server --model dist/hy3-demolition-mlx-lite-v1-fused --port 8080
```

## Rebuild REAP Artifact

```bash
./scripts/04_stream_calibrate_hy3_mlx.py \
  --model models/hy3-mlx-base-ar \
  --prompts eval/coding/prompts.jsonl \
  --soul-prompts eval/souls/protected_prompts.jsonl \
  --out dist/hy3-reap-saliency.json

./scripts/05_apply_reap_prune_hy3_mlx.py \
  --model models/hy3-mlx-base-ar \
  --saliency dist/hy3-reap-saliency.json \
  --out dist/hy3-reap-pruned \
  --ratio 0.25 \
  --write

./scripts/06_stream_requantize_hy3_mlx.py \
  --model dist/hy3-reap-pruned \
  --out dist/hy3-reap-requant \
  --write
```

Then heal and fuse as above.

Pruning is soul-preserving by default. To prevent accidental damage to
low-frequency skills, `05_apply_reap_prune_hy3_mlx.py` expects soul saliency for
coding, math, science, security, design, fullstack, gamedev, legacy, music,
art, and perfumery.
