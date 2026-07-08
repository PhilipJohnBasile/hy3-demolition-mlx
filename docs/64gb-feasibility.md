# Can Hy3 run on a 64 GB Mac? — researched verdict (2026-07-08)

> **UPDATE 2026-07-08 — PROVEN, not just theorized.** The SSD expert-streaming
> pager (`src/hy3_streaming.py`, `scripts/41_streaming_load.py`) now RUNS the
> full 295B reap25 hy_v3 model with **9.7 GB resident load, ~23 GB peak during
> generation, 3.85 tok/s, coherent output** — experts streamed from disk,
> bit-identical (zero quality loss). Hy3 fits a 64 GB Mac. The "must be built"
> path below is built and working (milestones M1-M6c).

Multi-agent research (8 agents: our Hy3 data + the GLM52 record + online SOTA,
adversarially fact-checked). Bottom line and the numbers below.

## Verdict

**A static, fully-resident Hy3 that fits 64 GB at good quality does not exist.**
The routed experts are **60.7 GB (86%)** of reap40 and already sit at the MLX
quant floor (gate/up 2-bit, down 3-bit, gs64) at ~0.528 GB/expert. A 64 GB Mac
gives ~50–54 GB usable. To fit, experts must drop 60.7 → ~40 GB = keep ~76/192
= **reap60 (60% prune)** — 20 points past the reap40 knee that already crashes
`brutal_lru_cache` (val-loss ladder 0.722→0.979→1.078, monotonic). The only
quality-safe lever (non-expert 8→4-bit) saves ~4.4 GB — it can't touch the 86%.
This is the same conclusion that cancelled BACKLOG #35.

## The one genuinely "still Hy3" path: SSD expert-streaming

Decouple resident RAM from the weight floor: keep attention + KV + router +
shared weights resident (~10 GB) plus an **LRU cache of hot experts**, and page
the cold top-8/192 experts from NVMe per token.
- **Fit:** ~15–45 GB resident (cache-tunable) — well under 54 GB.
- **Quality:** none lost — bit-identical Hy3 (could even stream unpruned lite-v1).
- **Cost:** speed — decode drops from 7.4 tok/s warm to **~2–5 tok/s** (SSD-bandwidth-bound: ~2.5 GB cold-expert read/token ÷ ~5 GB/s NVMe).
- **MLX-available? NO — must be built.** Stock `mlx_lm.server` eager-loads all
  experts and OOMs (mlx-lm issue #1438 is the same *class*). An explicit
  app-level MoE pager is required — do NOT rely on OS swap (Metal wired buffers
  + poor MoE page locality → swap-death-spiral/kernel panic).
- **Evidence the class works on Apple Silicon:** `deepseek-v4-flash-mlx` runs a
  100B MoE on a 48 GB Mac at ~4.5–5 tok/s — but it's third-party and
  **unreproduced for the hy_v3 architecture**. Existence-proof, not a guarantee.

## Best "good 64 GB experience" (relax "this exact model")

A genuinely smaller model, not a compression of this checkpoint:
- **35B-class MoE sibling** (MLX-native, fast, fits 64 GB *and* 32 GB with
  margin) — different model, same pipeline/soul DNA. Fastest route to a good
  64 GB experience today.
- **Distilled Hy3-lineage Mini** — REAP 34–40% + a *real distillation heal*
  (cloud multi-GPU, billions of tokens) rather than the light LoRA. Might make a
  deep prune hold quality. Not bit-identical Hy3; needs off-Mac training.
- There is **no official smaller Hy3** (Tencent ships only Hy3 295B + Hy3-FP8;
  Hunyuan-A13B is a different, weaker, older arch).

## Do NOT pursue

Deeper static pruning (reap50/60 — past the cliff), sub-2-bit codebook quant
(QTIP/VPTQ/AQLM/QuIP# — right idea, **no Metal kernel on MLX**, hard 2-bit
floor), 4-bit/mxfp4 requant (grows the model), expert merging (REAP's paper
shows it loses to pruning), CPU-RAM expert offload (a no-op on unified memory).

## Adversarial corrections (folded in)

Core arithmetic and conclusion CONFIRMED against committed receipts. Trimmed
over-cited externals: the "TurboQuant 122B-on-16GB bit-identical" demo is
unverified (dropped); mlx-lm #1438 is the same class (395 GB GLM on 128 GB),
not literally the 64 GB-Hy3 case; deepseek-v4-flash numbers are a different
architecture. No technique was falsely claimed to preserve quality below 2 bits.

## Recommendation

1. **Today, low-risk:** ship reap25 as the 96–128 GB daily driver (honestly not
   a 64 GB model); for 64 GB users, point them at the 35B sibling.
2. **The one novel bet worth trying:** build an SSD expert-streaming MoE pager
   for hy_v3 on MLX — the *only* route that keeps it literally Hy3 on 64 GB,
   zero quality loss, at ~2–5 tok/s. Scope it as a serving-engine project.
3. **If a *fast* small Hy3 is the goal:** distilled Mini, on rented GPUs.
