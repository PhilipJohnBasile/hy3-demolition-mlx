# Soul Protection Prompts

These prompts are not normal accuracy evals. They are REAP calibration guards.

Before pruning, run calibration with `eval/souls/protected_prompts.jsonl` so the
router saliency record contains per-soul expert usage. The pruning planner then
reserves each soul's top routed experts per layer before applying aggregate REAP.

Default protected souls:

- coding
- math
- science
- security
- design
- fullstack
- gamedev
- legacy
- music
- art
- perfumery
