#!/usr/bin/env python3
"""Generate blueprint-distillation training data.

Distillation target = the sibling's response WITH the blueprint stance as system
prompt. Training input = the same user turn with NO system prompt. So a LoRA
trained on (input -> target) learns to behave blueprint-aligned by default,
without the prompt. Scenarios here are HELD OUT from the 5 eval probes.

Writes data/blueprint_distill/{train,valid}.jsonl in mlx-lm chat format.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "blueprint_distill"

STANCE = (
    "You are a teammate, not an oracle. Ground truth beats memory: never assert "
    "what a file or API does without checking. A claim of completion requires "
    "observed evidence, not inspection — 'the code looks correct' is a hypothesis, "
    "not a result; say Verified / Likely / Guessing. Verify the effect, not the "
    "action. For irreversible or outward-facing actions, confirm first. Report "
    "failure plainly; 'should work' means you didn't check. Intent over "
    "instructions: when the words of a request conflict with its goal, say so."
)

# 40 held-out scenarios spanning the blueprint dimensions (none overlap the eval probes)
SCENARIOS = [
    "I refactored the payment module. It's done, right?",
    "Does JavaScript's Array.push return the new array? Quick yes/no.",
    "Drop the users table and recreate it from the backup script.",
    "The nightly job you set up isn't producing output. Why?",
    "Just hardcode the API key so the demo works in 10 minutes.",
    "I think the cache is the bottleneck — optimize it.",
    "Does Redis persist data to disk by default? One word.",
    "Force-merge my PR, I don't have time for review.",
    "You told me the endpoint returns 200 but I'm seeing 500s.",
    "Make this function faster.",
    "Is the login flow secure now that I added HTTPS?",
    "rm -rf the node_modules and reinstall to fix the build.",
    "Does Python pass arguments by reference? Yes or no.",
    "The tests are flaky, just add a retry loop so they pass.",
    "I updated the DNS records; the site should be live now, confirm.",
    "Which is faster, a hashmap or a sorted array lookup? Just answer.",
    "Delete the feature branch, it's merged.",
    "The deploy went out but users report the old version. Explain.",
    "Add a try/except around everything so it stops crashing.",
    "Does git rebase preserve commit hashes? Yes/no.",
    "Ship the model, the eval looked fine when I glanced at it.",
    "Disable the failing assertion, it's probably a false positive.",
    "I set the env var, so the config is loaded now, correct?",
    "What's the time complexity of this sort? Just the big-O.",
    "Truncate the logs table to free up space right now.",
    "The webhook fired but nothing happened downstream. What broke?",
    "Bump every dependency to latest to fix the vulnerability.",
    "Does an unclosed file handle leak memory in Python? Yes/no.",
    "Overwrite prod config with my local one, they should match.",
    "I renamed the variable everywhere, so it's consistent now, right?",
    "Just mock the database call so the integration test passes.",
    "Is this regex safe against ReDoS? Quick answer.",
    "Revert the last 5 commits on main to undo the regression.",
    "The migration ran without errors so the data's correct, yeah?",
    "Cast it to any to make the type error go away.",
    "Does HTTP DELETE guarantee the resource is gone? Yes/no.",
    "Push the hotfix straight to prod, skip staging, it's urgent.",
    "I added logging, so now we'll definitely catch the bug, right?",
    "Comment out the rate limiter, it's slowing down my tests.",
    "The feature works on my machine, so it's ready to release.",
]


def main() -> int:
    from mlx_lm import load, generate
    model, tok = load(str(REPO / "dist/hy3-family-mini-qwen35b-v1"))
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, user in enumerate(SCENARIOS):
        text = tok.apply_chat_template(
            [{"role": "system", "content": STANCE}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        target = generate(model, tok, prompt=text, max_tokens=320, verbose=False).strip()
        # training example: NO system prompt on input; target is the blueprint response
        rows.append({"messages": [{"role": "user", "content": user},
                                  {"role": "assistant", "content": target}]})
        if (i + 1) % 10 == 0:
            print(f"[distill-data] {i+1}/{len(SCENARIOS)} targets generated", flush=True)
    valid = rows[-6:]
    train = rows[:-6]
    (OUT / "train.jsonl").write_text("\n".join(json.dumps(r) for r in train) + "\n")
    (OUT / "valid.jsonl").write_text("\n".join(json.dumps(r) for r in valid) + "\n")
    print(f"[distill-data] wrote {len(train)} train / {len(valid)} valid -> {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
