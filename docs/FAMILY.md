# The Hy3-Demolition family

One project, a full family of local Apple-Silicon agent models — from "run the
real 295B on a 16 GB MacBook" to "a fast daily agent that fits 32 GB." Every
model is verifier-healed, receipt-backed, and Apache-2.0.

## Pick your model

| Want… | Use | Runs on | Speed |
|---|---|---|---|
| **A fast daily driver / agent** | **sibling** (Qwen35B + our heal) | 32–64 GB | **100+ tok/s** |
| The pruned Hy3 daily driver | reap25 | 96–128 GB | 7.4 tok/s |
| The unpruned Hy3 reference | lite-v1 | 128 GB | 7.4 tok/s |
| The real 295B Hy3 on *any* Mac | reap25 / lite-v1 via the **streaming pager** | **16–128 GB** | 0.7–3.85 tok/s |
| MTP speculative decoding (MTPLX) | any `*-mtp` variant | (sibling runs today) | — |

Manual pass (`docs/manual-pass.md`): **the sibling is the recommended default
for daily use** — equal quality to reap25, 6–10× faster. The Hy3 tiers are for
"I want the 295B's answer" or "run it on a tiny Mac."

## Published artifacts (Hugging Face, `philipjohnbasile/`)

| Repo | What | mtp/mtplx |
|---|---|---|
| `hy3-demolition-mlx-lite-v1` | unpruned Hy3, 104 GB | `-mtp` (backend-pending) |
| `hy3-demolition-mlx-reap25-v1` | 25% soul-preserving prune + heal, 80 GB | `-mtp` (backend-pending) |
| `hy3-family-mini-qwen35b-v1` | fast sibling, 18 GB, 8/10 stress | `-mtp` (MTPLX-recognized; runtime load pending) |

(reap40 = the rejected 40%-prune dead-end, kept private as evidence.)

## The three ideas that made it work

1. **Soul-preserving REAP** — prune experts by `gate × ‖output‖` mean, protecting
   rare "soul" experts. Knee at 25% (reap40 crashed a real code case).
2. **SSD expert-streaming pager** (`src/hy3_streaming.py`) — keep ~10 GB of
   non-expert weights resident + an LRU of hot experts, page the rest from disk
   per token. Bit-identical, so a 295B model runs on a 16 GB Mac at zero quality
   loss (just slow). This is the general answer to "too big for this Mac."
3. **Verifier-first heal + a fast sibling** — the same verifier-filtered SFT
   that healed Hy3 also turns a clean 35B MoE into a fast daily agent; the
   agent's outputs are validated by deterministic verifiers, so throughput +
   guardrails beat raw model size.

## Runtimes

- **Today:** `mlx_lm` (generate / chat / server) on the pinned fork
  (`eauchs/mlx-lm@hy_v3-mtp`); the sibling runs on stock mlx-lm.
- **Pending [ml-explore/mlx-lm#1211](https://github.com/ml-explore/mlx-lm/pull/1211):**
  puts `hy_v3` in mainline mlx-lm → unlocks **LM Studio** for the Hy3 AR models.
- **MTPLX** (speculative decoding): all `-mtp` variants are `forge probe`-recognized
  but not yet verified-runnable — MTPLX's bundled mlx_lm doesn't load
  `qwen3_5_mtp`/`hy_v3_mtp` yet. The Hy3 variants also wait on our backend PR
  ([youssofal/MTPLX#142](https://github.com/youssofal/MTPLX/pull/142)).

Full recipe, scripts, and a receipt for every number: this repo.
