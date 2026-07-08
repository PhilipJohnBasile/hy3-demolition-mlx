# Architecture

How Hy3-Demolition-MLX is put together, for someone reading the repo cold.
The one-line mental model: **a factory that turns source repos into a single
fused MLX model directory, with a verifier gate on everything that enters and
a receipt for every claim.**

## The invariant that explains everything

> The runtime dependency is the fused model directory plus `mlx_lm`. Nothing
> else. Every other repo, script, verifier, and dataset is *build-time only*.

If any step ever seems to require a runtime import to work, that is a defect in
the step, not a new dependency. This is why the eval harness, soul verifiers,
and demolition scripts never ship inside the model.

## Two products, one factory

- **lite-v1** (done): base Hy3 MLX checkpoint + a LoRA trained on
  verifier-filtered agent data, streamed-fused into a standalone directory. No
  pruning. ~104 GB, the current daily driver.
- **reap25** (next): lite's recipe plus soul-preserving 25% expert pruning +
  heal. ~80 GB target. Promoted only if it matches lite-v1 on the eval gate.
- **reap40** (conditional): 40% prune, only if reap25 wins cleanly. ~65 GB.

## Data flow (build time)

```
  source repos                     gates                       artifact
  ------------                     -----                       --------
  agent-brain-blueprint  ─┐
  agent-toolkit canons   ─┤   verifier mesh (agent-toolkit)
  glm52 verified fixes   ─┼──▶ + local verifiers  ──▶ SFT pack ──▶ LoRA ──┐
  tinygpt-souls          ─┘     (every row must pass)                     │
                                                                          ▼
  soul protected prompts ───▶ REAP calibration ──▶ prune plan ──▶ pruned ─▶ fuse
                              (true criterion,      (analyzer      weights   │
                               soul-bucketed)        ACCEPT/gate)            ▼
                                                                    fused MLX dir
                                                                    (mlx_lm serve)
```

- **Behavior** (how the agent acts) comes from `agent-brain-blueprint` +
  `tinygpt-souls`, distilled into the SFT stance/soul rows.
- **Expertise** (domain knowledge) comes from the `agent-toolkit` soul canons.
- **Verified repair data** (the training backbone) comes from the GLM run.
- **The verifier mesh** gates all of it: code is compiled + run, JSON parsed,
  tool-calls shape-checked, so only passing examples reach the weights.

## The pipeline, by script number

| Stage | Scripts | What |
|---|---|---|
| Base | `01`–`02`, `13` | download checkpoint, AR-only view (drop MTP sidecar) |
| Lite data | `11`, `15`, `16`, `21`, `22`, `31` | seed + import + length-normalize + canon + merge + audit |
| Lite train | `17` (train view), `07` (LoRA), `18` (streamed fuse) | the EOS-safe train/fuse path |
| Eval | `09` (runner), `20` (compare gate), `src/hy3_local_verifiers` | 30-case suite + hard + brutal tiers |
| REAP | `26` (calib pack), `04` (calibrate), `05` (prune), `25` (analyze), `06` (optional requant) | soul-preserving demolition |
| Serve | `08`, `14`, `mlx_lm.chat/server` | OpenAI-compatible local serving |
| Ops | `27` (baseline chain), `28` (overnight), `29` (preflight), `30` (benchmark) | automation + self-checks |

## Three hard-won invariants (see BUILD_NOTES for the scars)

1. **Train against the `is_training` template view** (`scripts/17`), never the
   plain AR view. Hy3's chat template only appends EOS to the final assistant
   turn when `is_training` is set; training without it produces a model that
   never stops.
2. **Stream everything bigger than half of RAM.** The stock `mlx_lm fuse`
   eager-loads 105 GB and dies; `scripts/18` fuses shard-by-shard at ~44 GB
   peak. Same principle for prune and requant.
3. **One model in memory at a time.** The checkpoint peaks ~112 GB of 128;
   GPU work is strictly serial, and heavy CPU/audit work must not run beside a
   resident model (it gets jetsam-killed — BUILD_NOTES 15).

## Governance (where decisions live)

- `BACKLOG.md` — numbered status board + execution order
- `DECISIONS.md` — the pre-committed rubric for every REAP/promotion call
- `BUILD_NOTES.md` — append-only incident log (what broke and why)
- `SECURITY.md` — runtime vs eval-harness trust model
- `RESTORE.md` — rebuild everything from source, including local patches
- `tests/` — 22 pure-logic regression tests (REAP, verifiers, gate, data prep)
- `eval/receipts/` — a JSON receipt behind every measured claim

## Runtime targets

The fused directory runs on stock `mlx_lm` today. Two other runtimes are being
enabled via upstream contributions (see `docs/upstream/`): **MTPLX** (a native
MTP speculative-decoding runtime — needs our `hy_v3` backend PR) and **LM
Studio** (needs `hy_v3` in stock mlx-lm). Both are the same fused directory,
different servers — the pure-model invariant holds across all three.
