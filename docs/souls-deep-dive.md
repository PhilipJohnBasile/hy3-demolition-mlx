# Soul-facet deep dive: sibling (healed) vs native Qwen base

**Date:** 2026-07-08 · **Models:** `hy3-family-mini-qwen35b-v1` (our verifier-healed
sibling) vs `mlx-community/Qwen3.6-35B-A3B-4bit` (unmodified base) · **Pack:**
`eval/souls/prompts.jsonl` (11 protected facets) · **Grader:** `verify_soul`
(keywords + substance + non-degeneracy) · **Mode:** `no_think`, greedy.

## Headline

**On the exact facets the heal trained on, the base is already as good as the
sibling — often producing near-identical text.** The verifier heal did *not* add
domain knowledge; the base Qwen3.6-35B already masters all 11 facets. Scores:

| | sibling | base |
|---|---|---|
| `verify_soul` pass rate | **10/11** | **11/11** |

The one sibling miss (gamedev) is a `max_tokens` truncation that the degeneracy
heuristic flagged — the visible content (game-loop + AABB collision code + a
summary table) is correct. So on-distribution, the honest read is **base ≥
sibling**: the heal is knowledge-neutral and cost one robustness glitch.

## The evidence: outputs are near-twins

The two models draw from the same base knowledge, often verbatim-close:

- **music** — both independently name the *same* "I–V–vi–IV (sensitive-female /
  axis) progression" and the *same* tension/release/voice-leading explanation.
  Base additionally cites "Let It Be / Someone Like You / With or Without You".
- **art** — both open with the identical "emotional architecture … light and
  color provide the soul" critique framing.
- **perfumery** — both build a correct citrus-woody pyramid from real materials
  (bergamot, yuzu, pink pepper …). Base is *more* thorough (names the scent,
  gives percentage blends).
- **coding / science / security / design / fullstack / legacy / math** — both
  pass on keywords + substance with comparable structure and depth.

## Per-facet

| facet | sibling | base | who's better | note |
|---|---|---|---|---|
| coding | PASS | PASS | tie | sibling code-first, base slightly more prose |
| math | PASS | PASS | tie | — |
| science | PASS | PASS | tie | — |
| security | PASS | PASS | sibling (longer, more concrete) | — |
| design | PASS | PASS | tie | — |
| fullstack | PASS | PASS | tie | near-identical length |
| gamedev | **FAIL** | PASS | base | sibling content fine; truncation → degeneracy flag |
| legacy | PASS | PASS | tie | — |
| music | PASS | PASS | tie | same progression, base adds song examples |
| art | PASS | PASS | tie | same framing |
| perfumery | PASS | PASS | base (more thorough) | base gives % blends |

## Stylistic difference (real, but not capability)

- **Sibling:** more direct/technical, code-first, emoji section headers, answers
  then stops.
- **Base:** more conversational/Socratic (gamedev ends "Where would you like to
  start our exploration?"), sometimes more thorough.

## What this means for the project

This is the on-distribution confirmation of the whole benchmark study's theme:
**the heal is ≈ neutral.** And crucially, it's neutral *not* because the eval is
off-distribution (souls is exactly what we trained on) but because **the base
model is already excellent at these domains** — a light LoRA heal has little room
to add capability the base lacks.

The project's real value therefore is **not** "a smarter model." It is:

1. **Local runnability** — REAP compression + the streaming pager put a
   295B-class model (and this fast 35B sibling) on consumer Macs.
2. **Consistent interface** — the `<|soul:facet|>` conditioning + format
   discipline as a *predictable* surface, not new knowledge.

The honest verdict — repeated across HumanEval, MBPP, AIME, GPQA, *and* the
on-distribution souls — is that the heal neither meaningfully helped nor hurt raw
quality. The base Qwen3.6 was already strong; we made it **local, compressed, and
consistently formatted**, not smarter.
