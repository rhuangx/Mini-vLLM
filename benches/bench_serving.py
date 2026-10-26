"""Throughput by concurrency: output tokens per second over a whole request set, against
transformers generate on the same left-padded batch.

    uv run python benches/bench_serving.py --concurrency 1 4 16 32
"""

from __future__ import annotations

import argparse
import gc
import time

import torch
from matplotlib.figure import Figure
from utils import engine, evidence, load_hf, random_prompts, save_plot, variants

from mini_vllm import DEFAULT_MODEL


def _hf_generate(model, tokenizer, prompts: list[list[int]], max_tokens: int) -> int:
    """Left-padded greedy generate. Returns the number of new tokens, one full length each."""
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    width = max(len(prompt) for prompt in prompts)
    pad = tokenizer.pad_token_id
    ids, mask = [], []
    for prompt in prompts:
        gap = width - len(prompt)
        ids.append([pad] * gap + list(prompt))
        mask.append([0] * gap + [1] * len(prompt))
    with torch.no_grad():
        out = model.generate(
            torch.tensor(ids, device=model.device),
            attention_mask=torch.tensor(mask, device=model.device),
            max_new_tokens=max_tokens,
            do_sample=False,
            eos_token_id=None,
        )
    torch.cuda.synchronize()
    return (out.shape[1] - width) * len(prompts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16, 32])
    parser.add_argument("--prompt-lens", type=int, nargs="+", default=[32, 128, 256, 512])
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()

    title = (f"Throughput: 2 x concurrency requests, prompts of {args.prompt_lens} tokens, "
             f"{args.max_tokens} output tokens each")
    with evidence(title):
        hf_model, tokenizer = load_hf(args.model)
        rows = {concurrency: {} for concurrency in args.concurrency}

        for label, backend in variants():
            for concurrency in args.concurrency:
                # Twice the requests there are slots, so finished requests are replaced mid-run.
                prompts = random_prompts(2 * concurrency, args.prompt_lens, len(tokenizer.vocab))
                llm = engine(hf_model, tokenizer, backend, num_blocks=2048, max_sequences=concurrency)
                print(f"{label}, concurrency {concurrency}", flush=True)
                start = time.perf_counter()
                completions = llm.generate(prompts, max_tokens=args.max_tokens)
                tokens = sum(len(c.token_ids) for c in completions)
                rows[concurrency][label] = tokens / (time.perf_counter() - start)
            del llm
        # The last engine's page pool is most of the 8 GB. Drop it before the baseline,
        # which keeps its own dense cache for the whole batch.
        gc.collect()
        torch.cuda.empty_cache()

        _hf_generate(hf_model, tokenizer.hf, [[1] * 32] * 2, 4)  # warm-up
        for concurrency in args.concurrency:
            prompts = random_prompts(2 * concurrency, args.prompt_lens, len(tokenizer.vocab))
            start = time.perf_counter()
            tokens = _hf_generate(hf_model, tokenizer.hf, prompts, args.max_tokens)
            rows[concurrency]["transformers generate"] = tokens / (time.perf_counter() - start)

        labels = list(rows[args.concurrency[0]])
        print("| concurrency | " + " | ".join(labels) + " |")
        print("|---" * (len(labels) + 1) + "|")
        for concurrency, row in rows.items():
            print(f"| {concurrency} | " + " | ".join(f"{row[label]:,.0f} tok/s" for label in labels) + " |")

        figure = Figure(figsize=(9, 5))
        figure.suptitle(title)
        axes = figure.subplots()
        for label in labels:
            throughputs = [rows[concurrency][label] for concurrency in args.concurrency]
            axes.plot(args.concurrency, throughputs, marker="o", label=label)
        axes.set_xscale("log", base=2)
        axes.set_xticks(args.concurrency, [str(concurrency) for concurrency in args.concurrency])
        axes.set_xlabel("concurrency")
        axes.set_ylabel("throughput (tok/s)")
        axes.legend()
        save_plot(figure, "bench_serving")


if __name__ == "__main__":
    main()
