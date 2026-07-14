"""Entrypoint: VALSE accuracy sweep over every configured instrument.

    uv run python run-valse-acc-vlm_decoders.py <task> <model> <save_json> <data_root>
        [--dtype float16|float32] [--decoding sample|greedy] [--seed N] [--deterministic]

Reports pairwise accuracy (acc_r) and caption-only / foil-only precision (p_c / p_f).
``<task>`` is accepted for CLI compatibility but ignored: the sweep runs every instrument
(except mscoco). See ``ccshapvlm.valse_eval.run_accuracy``.
"""

from __future__ import annotations

import argparse
import time

from ccshapvlm.cli import add_runtime_flags, apply_runtime_args, bootstrap_runtime
from ccshapvlm_repro import load_env


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("task", help="ignored; the sweep runs every instrument")
    parser.add_argument("model")
    parser.add_argument("save_json", type=int)
    parser.add_argument("data_root")
    add_runtime_flags(parser)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    apply_runtime_args(args)
    load_env()
    bootstrap_runtime()

    from ccshapvlm.models import load_adapter
    from ccshapvlm.valse_eval import run_accuracy

    start = time.time()
    adapter = load_adapter(args.model)
    run_accuracy(adapter, args.model, args.data_root, write=bool(args.save_json))
    print(f"\nThis script ran for {time.time() - start:.2f}s.")


if __name__ == "__main__":
    main()
