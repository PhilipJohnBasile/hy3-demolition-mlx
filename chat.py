#!/usr/bin/env python3
"""Chat with Hy3 on Apple Silicon via MLX, with reasoning-effort control.

Usage:
    . .venv/bin/activate
    python chat.py                    # no_think (direct response)
    python chat.py --effort high      # chain-of-thought (math/coding/reasoning)
    python chat.py --effort low       # light reasoning
"""
import argparse
import sys

from mlx_lm import load, generate

MODEL = "ox-ox/Hy3-295B-Instruct-w2q3exp-AProjQ8-SExpQ8-OutQ8-MTP-mlx"


def main():
    p = argparse.ArgumentParser(description="Chat with Hy3 via MLX")
    p.add_argument("--model", default=MODEL, help="HF model id")
    p.add_argument("--effort", choices=["no_think", "low", "high"],
                   default="no_think", help="reasoning_effort")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temp", type=float, default=0.9, help="temp (Hy3 recommends 0.9)")
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--prompt", help="single prompt; omit for interactive REPL")
    args = p.parse_args()

    print(f"Loading {args.model} ... (first run downloads ~110GB)", file=sys.stderr)
    model, tokenizer = load(args.model)
    print(f"Loaded. reasoning_effort={args.effort}", file=sys.stderr)

    def ask(prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            chat_template_kwargs={"reasoning_effort": args.effort},
        )
        return generate(
            model, tokenizer, prompt=formatted,
            max_tokens=args.max_tokens, temp=args.temp, top_p=args.top_p,
        )

    if args.prompt:
        print(ask(args.prompt))
        return

    print("Interactive mode. Ctrl-D or type /exit to quit.\n", file=sys.stderr)
    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user or user in ("/exit", "/quit"):
            break
        print("hy3> " + ask(user) + "\n")


if __name__ == "__main__":
    main()
