"""Shared plumbing for the ``run-*`` entrypoints.

Every entrypoint exposes the same runtime flags (``--dtype/--decoding/--scoring/--seed/
--deterministic``) and the same start-up dance (silence library logging, free CUDA memory,
seed, honor determinism). Keeping both here means the scripts stay thin and identical where
they should be, differing only in their positional arguments and which experiment they call.
"""

from __future__ import annotations

import argparse

from ccshapvlm_repro import apply_runtime_flags


def add_runtime_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dtype", choices=["float16", "float32"],
                        help="model + image precision (default: config.DTYPE)")
    parser.add_argument("--decoding", choices=["sample", "greedy"],
                        help="answer/explanation decoding (default: config.DECODING)")
    parser.add_argument("--scoring", choices=["efficient", "teacher_forcing"],
                        help="SHAP target scoring: first-step logits vs teacher-forced "
                             "log-odds (default: config.SCORING)")
    parser.add_argument("--seed", type=int, help="seed numpy/torch for reproducible runs")
    parser.add_argument("--deterministic", action="store_true",
                        help="force deterministic CUDA kernels (use with a single GPU)")


def apply_runtime_args(args: argparse.Namespace) -> None:
    apply_runtime_flags(args.dtype, args.decoding, args.seed, args.deterministic, args.scoring)


def bootstrap_runtime() -> None:
    """Prepare the process for a run. Import-heavy; call only after ``load_env``."""
    import logging
    import random

    import torch
    from accelerate import Accelerator
    from transformers.utils import logging as hf_logging

    from ccshapvlm_repro import maybe_enable_determinism, maybe_seed

    print("Cuda is available:", torch.cuda.is_available())
    torch.cuda.empty_cache()
    Accelerator().free_memory()
    hf_logging.set_verbosity_error()
    logging.getLogger("shap").setLevel(logging.ERROR)

    random.seed(42)
    maybe_seed()  # override with CCVLM_SEED when the regression harness sets it
    maybe_enable_determinism()
