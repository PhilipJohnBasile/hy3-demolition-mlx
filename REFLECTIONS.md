# Reflections

Cross-project learnings and an honest assessment of where Hy3-Demolition-MLX
actually stands. This is the "what do we think" doc — opinions and judgment,
not receipts. Written 2026-07-07.

## Learnings from the perfume agent (a sibling MLX project)

The perfume agent (`perfume-soul`, private) runs `Qwen3.6-35B-A3B-4bit`
in-process on MLX with a custom HTTP server and per-session memory. It is
architecturally the **opposite** of Hy3, and the contrast is instructive.

| | Perfume agent | Hy3-Demolition-MLX |
|---|---|---|
| Serving | custom in-process HTTP handler | stock `mlx_lm.server` (OpenAI API) |
| State | per-session memory: history + formula ledger + constraints | stateless; client sends full history |
| Domain rules | enforced at runtime in the wrapper (editable) | baked into the weights (permanent, wrapper-free) |
| Right for | a specialized single-domain agent where the running state IS the product | a general daily-driver any tool can point at |

**Both are correct — for different products.** The lesson is not to converge
them; it is that we now own a proven template for *each* pattern.

Concretely reusable from the perfume `serve.py`:

- **Client-generated session id** (`crypto.randomUUID()` → `sessionStorage` →
  `sid` in the request body, server clamps to 64 chars). No server-side
  session allocation, no cookies. Clean.
- **LRU session cap with a lock** (`_SESS_CAP`, pop-and-reinsert to mark MRU,
  evict oldest past the cap) — bounded memory for concurrent users.
- **Token auth** (`X-Brain-Token`) — Hy3's server has none; the minimum if it
  is ever exposed past localhost.
- **A bug worth remembering**: the handler read `rfile.read(n)` twice (once
  for `query`, once for `body`); the second read gets an empty stream because
  it is already consumed. Read the body once, then `.get()` fields. (Hy3 is
  unaffected — we use stock `mlx_lm.server`, not a custom handler.)

**The load-bearing data point**: a 35B MoE runs comfortably in-process on MLX
and serves fast. That is direct evidence the **shelved 32 GB Hy3-Mini tier is
feasible** — a small domain-specialized Hy3 variant would be served by exactly
this pattern. The proof-of-concept and the serving code already exist.

**The constraint tradeoff worth being conscious of**: perfume keeps domain
rules in the session (runtime-editable without retraining); Hy3 bakes them
into the weights (permanent, no wrapper). For a specialized nose-agent,
runtime-editable constraints are probably *better* than baking them in. For a
general daily driver, wrapper-free is better. Pick per product, deliberately.

## Honest assessment of what we're doing now

The good, stated plainly, then the concerns — because cheerleading is not
useful and the receipts culture demands candor.

### What is genuinely strong

- **Receipts-first actually works.** It caught the EOS bug, the truncation
  bug, the promotion-gate bug, and the requant corruption — none by intuition,
  all by verification. This is the project's real moat.
- **Audit-before-GPU paid for itself repeatedly.** Reading code and testing on
  tiny models caught bugs that would have cost GPU-days on the 295B.
- **The decision rubric (DECISIONS.md) front-loads judgment** so the promotion
  calls are mechanical, not improvised under pressure.
- **lite-v1 is a real, shipped, honest artifact.** Whatever happens with REAP,
  there is a promoted model with receipts for every claim.

### What I would flag as real risks (in priority order)

1. **We have not validated that our eval instrument can measure pruning
   damage.** lite-v1 scored ~perfect on the 30-case suite and 8/8 hard tier.
   The brutal tier was built to discriminate but has NEVER been run against
   the model. If lite-v1 also scores ~perfect on brutal, we have no measuring
   tool, and the entire reap25 promotion decision rests on an instrument we
   cannot yet prove works. **The fresh baseline (esp. brutal) is not just a
   gate step — it is the validation that the gate is meaningful. Run it and
   read the failures before trusting any REAP comparison.**

2. **Is reap25 worth it?** The paper says a 25% prune costs ~2.8% on codegen.
   reap25 (~80 GB) vs lite-v1 (~104 GB) both run on the 128 GB machine; the
   win is ~24 GB of headroom, not a capability or speed gain (decode is
   unchanged). That headroom is real (IDE + browser + model coexist) but it is
   a "nicer daily driver," not a breakthrough. We should be willing to
   conclude **lite-v1 is the product and reap25 is optional** if the quality
   cost does not justify the headroom.

3. **The base model may already be most of lite-v1.** The partial base
   baseline (23/24, failing only the fabricated-execution planning case)
   suggests the LoRA's value is **honesty and stop-discipline, not raw
   capability** — the base was already strong. That is a legitimate and useful
   finding, but it reframes the project: the *demolition* (prune) is where
   capability is at stake; the *heal* is mostly behavioral polish. Do not
   over-invest the heal expecting it to restore capability the prune removes.

4. **The speed story is entirely external.** MTP self-speculation was 13.7×
   slower in mlx-lm. The only speed lever left is MTPLX's batched verify —
   which depends on our unmerged PR, an unmerged mlx-lm base PR, and a
   maintainer's time. If none land, the model is 7.4 tok/s warm forever. That
   is a usable daily driver, but the "speed plus brains" goal is currently
   half-delivered and the other half is not in our control.

5. **Scope is expanding faster than the core need.** The core deliverable — a
   good local agent model — is essentially done (lite-v1). Family tiers,
   benchmarks, the MTPLX contribution, GGUF (cancelled), a 32 GB Mini: these
   are distribution and refinement. All legitimate, but worth naming as
   *expansion*, not *completion of the original goal*, so effort is spent
   deliberately.

### What I actually think the next moves should be

1. **The manual quality pass (#16) is the highest-value unrun step, full
   stop.** Every receipt says lite-v1 works; none can say whether it is
   *pleasant to use*. Drive it on real code, planning, tool-use, and a soul
   task. That single session determines whether the project has produced
   something worth using — and it is the one thing no automation can answer.
2. **Run the fresh baseline to validate the instrument** (concern #1) before
   committing GPU-days to reap25. If brutal doesn't discriminate on lite-v1,
   fix the instrument first.
3. **Hold reap25 to an honest bar**: promote only if it is genuinely a better
   daily driver, and be willing to keep lite-v1 as the product if not.
4. **Treat the MTPLX/speed track as a bonus, not a dependency.** Ship on AR
   speed; if MTPLX lands, it is upside.

The short version: **we have built something real and verified it honestly.
The biggest open question is not technical execution — it is whether the next
increments (reap25, the family, the speed lever) are worth their cost, and the
cheapest way to find out is the manual pass plus one honest baseline run.**
