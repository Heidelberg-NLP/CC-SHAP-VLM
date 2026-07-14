"""Entrypoint: run VLM faithfulness / CC-SHAP experiments for one task and model.

    uv run python run-faithfulness.py <task> <model> <num_samples> <save_json> <data_root>
        [--dtype float16|float32] [--decoding sample|greedy] [--seed N] [--deterministic]

``--dtype``/``--decoding`` default to ``ccshapvlm.config``; the regression harness passes
``--decoding greedy --seed N`` for reproducibility. ``load_env`` must run before importing
transformers, so the heavy imports follow argument parsing.
"""

from __future__ import annotations

import argparse
import time

from ccshapvlm.cli import add_runtime_flags, apply_runtime_args, bootstrap_runtime
from ccshapvlm_repro import load_env


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("task")
    parser.add_argument("model")
    parser.add_argument("num_samples", type=int)
    parser.add_argument("save_json", type=int)
    parser.add_argument("data_root")
    add_runtime_flags(parser)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    apply_runtime_args(args)
    load_env()  # set HF token + cache dir before importing transformers/huggingface_hub
    bootstrap_runtime()

    from ccshapvlm.evaluation import run
    from ccshapvlm.models import load_adapter

    start = time.time()
    adapter = load_adapter(args.model)
    run(adapter, args.task, args.model, args.num_samples, args.data_root,
        write=bool(args.save_json))
    print(f"\nThis script ran for {time.time() - start:.2f}s.")


if __name__ == "__main__":
    main()
