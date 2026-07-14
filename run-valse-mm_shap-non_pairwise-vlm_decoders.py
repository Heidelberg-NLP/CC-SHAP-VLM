"""Entrypoint: non-pairwise VALSE MM-SHAP for a single instrument.

    uv run python run-valse-mm_shap-non_pairwise-vlm_decoders.py <task> <model> \\
        <num_samples> <data_root> [--dtype ...] [--decoding ...] [--seed N] [--deterministic]

Scores the caption and the foil separately (rather than as one A/B question) and reports
their precision (p_c / p_f) and MM-SHAP (T-SHAP_c / T-SHAP_f). See
``ccshapvlm.valse_eval.run_non_pairwise``.
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
    parser.add_argument("data_root")
    add_runtime_flags(parser)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    apply_runtime_args(args)
    load_env()
    bootstrap_runtime()

    from ccshapvlm.models import load_adapter
    from ccshapvlm.valse_eval import run_non_pairwise

    start = time.time()
    adapter = load_adapter(args.model)
    run_non_pairwise(adapter, args.task, args.model, args.data_root,
                     num_samples=args.num_samples or None)
    print(f"\nThis script ran for {time.time() - start:.2f}s.")


if __name__ == "__main__":
    main()
