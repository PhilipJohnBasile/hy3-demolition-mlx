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
  - mtp
  - speculative-decoding
  - mtplx
  - qwen3
---

# Hy3-Family Mini — Qwen35B MTP v1

> ⚠️ MTP variant is **staged/recognized, not yet verified-runnable** — see Status below. For inference today use the AR sibling.

The **MTP-equipped** variant of [hy3-family-mini-qwen35b-v1](https://huggingface.co/philipjohnbasile/hy3-family-mini-qwen35b-v1): our verifier-healed Qwen3.6-35B-A3B trunk with the Qwen **NextN / MTP head** (`mtp.*`, from the clean `mlx-community/Qwen3.6-35B-A3B-MTP-4bit`) grafted on for self-speculative decoding.

## Status — MTP-recognized, runtime load pending (honest)

`mtplx forge probe` recognizes this artifact (`has_mtp_weights: true`, `forgeable: true`, backend `qwen3_next`, ~25 GB peak). **However**, `mtplx forge build` currently fails to *load* it — MTPLX's bundled `mlx_lm` raises `Model type qwen3_5_mtp not supported`. So, like the Hy3 MTP variants, this is **staged, not yet verified-runnable**: the MTP weights and metadata are correct and recognized, but the runtime load path for `qwen3_5_mtp` isn't there yet in the release. Use the [AR sibling](https://huggingface.co/philipjohnbasile/hy3-family-mini-qwen35b-v1) for actual inference today.

```bash
mtplx forge build --repo <this-repo> --recipe '{"body_bits":4,"body_group_size":64,"body_mode":"affine","mtp_policy":"keep_bf16"}' ...
mtplx tune --depths 1,2,3     # find the best speculative depth for your Mac
```

## What it is
- Trunk: `mlx-community/Qwen3.6-35B-A3B-4bit` + our verifier-filtered LoRA heal (val 0.639, 8/10 executed stress — see the AR sibling card).
- MTP head: the clean Qwen NextN head (a draft component MTPLX verifies every token against, so it never affects correctness — only draft acceptance/speed).
- Clean base — NOT the perfume-private Mimosa build.

## Notes
- The fast AR daily-driver path is the [AR sibling](https://huggingface.co/philipjohnbasile/hy3-family-mini-qwen35b-v1); this variant adds MTP speculative decoding on MTPLX.
- Graft script + receipts: https://github.com/PhilipJohnBasile/hy3-demolition-mlx
