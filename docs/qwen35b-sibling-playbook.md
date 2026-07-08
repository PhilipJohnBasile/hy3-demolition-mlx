# 35B sibling playbook — mlx-community/Qwen3.6-35B-A3B-4bit (BACKLOG #38)

PJB-provided playbook for taking the clean Qwen base into a Hy3-family sibling
(the fast 64GB/32GB tier). Base chosen: **mlx-community/Qwen3.6-35B-A3B-4bit**
(~20 GB, MLX-native, qwen3_5_moe). NOT the perfume-private Mimosa build.

## 1. Download
```
hf download mlx-community/Qwen3.6-35B-A3B-4bit --local-dir models/qwen35b-a3b-base
```

## 2. Sanity check
```
python -m mlx_lm.generate --model models/qwen35b-a3b-base \
  --prompt "Explain X in three sentences." --max-tokens 300
```
**Qwen3.6 quirk (bit us on Mimosa):** to turn OFF thinking, pass
`enable_thinking=False` as a DIRECT kwarg to `apply_chat_template` — the nested
`chat_template_kwargs` form and `/no_think` are ignored.

## 3. Customize (cheapest first)
- **A) Prompt/behavior only** — strong system prompt + our tools/verifiers. Zero
  GPU. This is how Mimosa uses it (model proposes, deterministic code validates).
- **B) LoRA fine-tune (the real customize)** — our verifier-filtered SFT pack as
  train/valid JSONL:
  ```
  python -m mlx_lm.lora --model models/qwen35b-a3b-base --train --data ./data \
    --iters 600 --batch-size 2 --num-layers 8 --adapter-path ./my-adapters
  python -m mlx_lm.fuse --model models/qwen35b-a3b-base \
    --adapter-path ./my-adapters --save-path ./my-custom-model
  ```
  LoRA on a 4-bit MoE fits comfortably on the M5 Max. Start ~few hundred
  examples, --num-layers 8, scale if underfit.
- **C) Re-quantize** for a different size: `mlx_lm.convert -q --q-bits 6 ...`

## 4. Serve
- In-process: `from mlx_lm import load` (fastest, private — Mimosa's way).
- API: `python -m mlx_lm.server --model ./my-custom-model --port 8080` (OpenAI /v1).
- **MTP speed (MTPLX path):** after fine-tuning,
  `mtplx forge build --repo <your-fused-model> --recipe '{"body_bits":4,"body_group_size":64,"body_mode":"affine","mtp_policy":"keep_bf16"}'`
  then `mtplx tune --depths 1,2,3`. Gives the customized model the ~25 turns/min serving.

## Two carry-overs from tonight
1. **Fine-tune the FUSED model, not the forged MTP one** — forge the MTP head
   AFTER fine-tuning, from the fused result. (This is #39.)
2. **Env-var the model path** (like Mimosa's `BRAIN_MODEL`) so swapping
   base / fused / MTP is a one-line change.

## Our adaptation notes
- Reuse our verifier-filtered SFT pack (`data/hy3_lite_sft_combined`) — but Qwen
  has a different tokenizer/chat template, so the `is_training` EOS trick and
  soul-tag format are Hy3-specific; verify Qwen's template appends EOS in training
  before trusting the heal (re-run the D2 stop-behavior smoke).
- Eval with the same harness (`scripts/09` + brutal/hard tiers) for apples-to-apples
  vs the Hy3 family.
- Keep it a SEPARATE artifact line (`hy3-family-mini-qwen35b-*`), clearly a
  different base — same soul/verifier DNA, not "small Hy3".
