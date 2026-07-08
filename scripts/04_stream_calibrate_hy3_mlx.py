#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

mlx_generate = importlib.import_module("mlx_lm.generate")


@contextlib.contextmanager
def no_wired_limit(*_args, **_kwargs):
    yield


@dataclass(frozen=True)
class PromptCase:
    prompt: str
    facet: str = "default"
    source: str = "calibration"


def load_prompts(path: str) -> list[PromptCase]:
    prompts: list[PromptCase] = []
    with Path(path).open() as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            prompts.append(
                PromptCase(
                    prompt=obj.get("prompt") or obj.get("content") or line.strip(),
                    facet=obj.get("facet") or obj.get("soul") or "default",
                    source=str(path),
                )
            )
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/hy3-mlx-base")
    parser.add_argument("--prompts", default="eval/coding/prompts.jsonl")
    parser.add_argument("--soul-prompts", default="eval/souls/protected_prompts.jsonl")
    parser.add_argument("--out", default="dist/hy3-reap-saliency.json")
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--reasoning-effort", default="no_think")
    parser.add_argument("--wired-limit", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=25,
                        help="save resumable checkpoint every N prompts")
    parser.add_argument("--max-prompt-tokens", type=int, default=1024,
                        help="truncate calibration prompts to this many tokens "
                        "(prefill of giant prompts dominates wall-clock; routing "
                        "saliency is captured well within ~1k tokens)")
    args = parser.parse_args()

    if not args.wired_limit:
        mlx_generate.wired_limit = no_wired_limit

    model, tokenizer = load(args.model, lazy=True)
    model.num_nextn_predict_layers = 0
    if hasattr(model, "mtp"):
        delattr(model, "mtp")

    def zero_bucket(n: int) -> dict:
        return {
            "counts": [0] * n,
            "score_sum": [0.0] * n,
            "reap_sum": [0.0] * n,
        }

    layers = {}
    moe_cls = None
    for i, layer in enumerate(model.layers):
        mlp = getattr(layer, "mlp", None)
        router = getattr(mlp, "router", None)
        if router is not None:
            setattr(mlp, "_hy3_reap_layer", i)
            moe_cls = mlp.__class__
            n = router.expert_bias.shape[0]
            layers[i] = {**zero_bucket(n), "facets": {}}

    if not layers or moe_cls is None:
        raise RuntimeError("no Hy3 MoE layers found")

    original_call = moe_cls.__call__
    active_facet = "default"

    # True REAP saliency (arXiv:2510.13999 eq. 9) needs gate value AND the
    # L2 norm of the selected expert's output: S_j = mean over routed tokens
    # of g_j(x) * ||f_j(x)||. Replicate the MoE forward once (a wrapped
    # re-call would double prefill cost) and accumulate both.
    def recording_call(self, x):
        if self.sharding_group is not None:
            return original_call(self, x)
        inds, scores = self.router(x)
        combine_scores = scores if self.fp32_combine else scores.astype(x.dtype)
        y = self.switch_mlp(x, inds)
        layer_idx = getattr(self, "_hy3_reap_layer", None)
        if layer_idx is not None:
            norms = mx.linalg.norm(y.astype(mx.float32), axis=-1)
            mx.eval(inds, scores, norms)
            idx_rows = inds.reshape(-1).tolist()
            score_rows = scores.reshape(-1).tolist()
            norm_rows = norms.reshape(-1).tolist()
            bucket = layers[layer_idx]
            facet_bucket = bucket["facets"].setdefault(
                active_facet, zero_bucket(len(bucket["counts"]))
            )
            for idx, score, norm in zip(idx_rows, score_rows, norm_rows):
                contribution = float(score) * float(norm)
                for b in (bucket, facet_bucket):
                    b["counts"][idx] += 1
                    b["score_sum"][idx] += float(score)
                    b["reap_sum"][idx] += contribution
        out = (y * combine_scores[..., None]).sum(axis=-2)
        if self.shared_mlp is not None:
            out = out + self.shared_mlp(x)
        return out.astype(x.dtype)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ckpt = out.with_suffix(".ckpt.json")

    def build_payload(done: int) -> dict:
        return {
            "model": args.model,
            "prompts": args.prompts,
            "soul_prompts": args.soul_prompts,
            "criterion": "reap_sum = sum g_j(x)*||f_j(x)||_2 over routed tokens; "
                         "divide by counts for the REAP mean (arXiv:2510.13999 eq. 9). "
                         "score_sum (gate-only) kept for comparison.",
            "prompts_processed": done,
            "created_at": time.time(),
            "layers": {str(k): v for k, v in sorted(layers.items())},
        }

    # Resume: a killed run (GPU needed elsewhere) leaves a checkpoint holding
    # the accumulators + how many prompts were already folded in. Reload them
    # and skip that many, so an interruption costs only the in-flight prompt.
    start_at = 0
    if ckpt.exists():
        try:
            prior = json.loads(ckpt.read_text())
            for k, v in prior.get("layers", {}).items():
                if int(k) in layers:
                    layers[int(k)] = v
            start_at = int(prior.get("prompts_processed", 0))
            print(f"resuming from checkpoint: {start_at} prompts already folded in")
        except (ValueError, KeyError) as e:
            print(f"checkpoint unreadable ({e}); starting fresh")

    moe_cls.__call__ = recording_call
    try:
        cases = load_prompts(args.prompts)
        soul_path = Path(args.soul_prompts)
        if soul_path.exists():
            cases.extend(load_prompts(str(soul_path)))
        for i, case in enumerate(cases):
            if i < start_at:
                continue
            active_facet = case.facet
            messages = [{"role": "user", "content": case.prompt}]
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                reasoning_effort=args.reasoning_effort,
            )
            prompt_tokens = tokenizer.encode(formatted, add_special_tokens=False)
            # Cap prompt length: the calibration pool has prompts up to ~16k
            # tokens whose prefill dominates wall-clock, while routing saliency
            # is fully exercised in the first ~1k tokens (REAP packs to 2048).
            # Truncating keeps calibration tractable without hurting the signal.
            if args.max_prompt_tokens and len(prompt_tokens) > args.max_prompt_tokens:
                prompt_tokens = prompt_tokens[: args.max_prompt_tokens]
            mlx_generate.generate(
                model,
                tokenizer,
                prompt=prompt_tokens,
                max_tokens=args.max_tokens,
                verbose=False,
            )
            if (i + 1) % args.checkpoint_every == 0:
                ckpt.write_text(json.dumps(build_payload(i + 1)))
                print(f"  checkpoint at prompt {i + 1}/{len(cases)}", flush=True)
    finally:
        moe_cls.__call__ = original_call

    out.write_text(json.dumps(build_payload(len(cases)), indent=2) + "\n")
    ckpt.unlink(missing_ok=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
