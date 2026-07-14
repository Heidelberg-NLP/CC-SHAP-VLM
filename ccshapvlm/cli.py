"""Shared plumbing for the ``run-*`` entrypoints.

Every entrypoint exposes the same runtime flags (``--dtype/--decoding/--scoring``) and the
same start-up dance (silence library logging, free CUDA memory). Keeping both here means
the scripts stay thin and identical where they should be, differing only in their
positional arguments and which experiment they call.
"""

from __future__ import annotations

import argparse


def add_runtime_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dtype", choices=["float16", "float32"],
                        help="model + image precision (default: config.DTYPE)")
    parser.add_argument("--decoding", choices=["sample", "greedy"],
                        help="answer/explanation decoding (default: config.DECODING)")
    parser.add_argument("--scoring", choices=["efficient", "teacher_forcing"],
                        help="SHAP target scoring: first-step logits vs teacher-forced "
                             "log-odds (default: config.SCORING)")


def apply_runtime_args(args: argparse.Namespace) -> None:
    """Override the config defaults with any runtime flags the user passed."""
    from ccshapvlm import config

    if args.dtype:
        config.DTYPE = args.dtype
    if args.decoding:
        config.DECODING = args.decoding
    if args.scoring:
        config.SCORING = args.scoring


def bootstrap_runtime() -> None:
    """Import-heavy (torch/accelerate/transformers): call after ``load_env`` so the HF
    cache/token env vars are exported before transformers is first imported."""
    import logging

    import torch
    from accelerate import Accelerator
    from transformers.utils import logging as hf_logging

    print("Cuda is available:", torch.cuda.is_available())
    torch.cuda.empty_cache()
    Accelerator().free_memory()
    hf_logging.set_verbosity_error()
    logging.getLogger("shap").setLevel(logging.ERROR)
