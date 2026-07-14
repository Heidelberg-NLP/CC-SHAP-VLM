"""Entrypoint: VALSE MM-SHAP sweep (pairwise accuracy + modality importance).

    uv run python run-valse-mm-shap-vlm_decoders.py <task> <model> <num_samples> \\
        <save_json> <data_root> [--dtype ...] [--decoding ...] [--scoring ...]

Reports pairwise accuracy (acc_r) and MM-SHAP (T-SHAP_r) over every configured instrument
(except mscoco). ``<task>`` is accepted for CLI compatibility but ignored; ``<num_samples>``
caps each instrument (omit / 0 processes all accepted items). See
``ccshapvlm.valse_eval.run_mm_shap``.
"""

from __future__ import annotations

import argparse
import time

from ccshapvlm.cli import add_runtime_flags, apply_runtime_args, bootstrap_runtime
from hf_env import load_env


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("task", help="ignored; the sweep runs every instrument")
    parser.add_argument("model")
    parser.add_argument("num_samples", type=int, help="per-instrument cap; 0 means all")
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
    from ccshapvlm.valse_eval import run_mm_shap

    start = time.time()
    adapter = load_adapter(args.model)
    run_mm_shap(adapter, args.model, args.data_root, num_samples=args.num_samples or None,
                write=bool(args.save_json))
    print(f"\nThis script ran for {time.time() - start:.2f}s.")


if __name__ == "__main__":
    main()
