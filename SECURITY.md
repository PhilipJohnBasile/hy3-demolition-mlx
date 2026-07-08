# Security Model

This project has two very different trust surfaces: the **runtime** (what you
get when you serve the model) and the **build/eval harness** (what turns
source repos into the model and scores it). They have opposite risk profiles.

## Runtime — the fused model (low risk)

The released artifact is a fused MLX safetensors directory served by
`mlx_lm.generate` / `mlx_lm.chat` / `mlx_lm.server`. Inference does **not**
execute anything the model emits — it produces text. There is no tool
execution, no shell, no filesystem or network authority inside the served
model (see the "pure model" rule in the README). Standard LLM caveats apply
(don't feed it secrets you wouldn't want echoed; treat its output as
untrusted input to downstream systems), but the runtime itself runs no code
on your behalf.

## Build / eval harness — executes model-generated code (handle with care)

The verifier mesh (`agent-toolkit`, invoked by `scripts/09_eval_agent_toolkit.py`
and the SFT data filters) works by **running** model-generated code to check
it compiles and passes tests. Concretely, for code domains it writes the
model's output to a temp file and runs `subprocess.run([python, file],
timeout=30)` in a `TemporaryDirectory`.

Isolation is therefore limited to:

- a throwaway working directory (protects the *cwd*, not your home/filesystem)
- a 30-second timeout (stops infinite loops, not fast malicious code)

It is **NOT** a sandbox. Model-generated code runs with the **full privileges
of the user** running the harness — it can read/write files outside the temp
dir, open network connections, spawn processes, etc. A prompt that induces the
model to emit `os.system(...)`, an exfiltration call, or a fork bomb would
have those effects.

### Why this is acceptable here

We only ever run the harness on:

- **our own** model (or the base checkpoint), not an adversary's, and
- **our own curated eval cases** (`eval/*/prompts.jsonl`) with fixed harnesses,
  on a personal machine.

The threat model for a *targeted* malicious model producing hostile code does
not apply to running your own model against your own tests.

### If you run this on untrusted output

If you point the eval harness at a model you don't control, or at arbitrary
generations, **sandbox it**: a container with no network, a locked-down user,
seccomp/nsjail, or a disposable VM. Do not run untrusted model code on a host
you care about just because the harness makes it convenient.

## Reporting

This is a private-author research project, not a supported product. Issues can
be filed on the GitHub repo. There is no security SLA.
