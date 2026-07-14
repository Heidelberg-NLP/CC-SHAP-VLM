"""Environment, determinism, and model-code helpers shared by CC-SHAP-VLM.

``load_env`` reads ``.env`` (HF_TOKEN, HF_CACHE) and points the HuggingFace token and
cache-location env vars at it. It must run before ``transformers``/``huggingface_hub`` are
imported, since those read ``HF_HOME`` and the token at import time.

``maybe_seed`` makes a run reproducible when ``CCVLM_SEED`` is set. It matters here because
generation uses ``do_sample=True``: without a seed the sampled explanation (and thus the
CC-SHAP score computed on it) differs run to run, so the before/after regression harness
sets it to compare the two dependency stacks on equal footing.

``ensure_mplug_remote_code`` overlays the patched ``modeling_mplugowl3.py`` (vendored under
``mplug_owl3/``, Apache-2.0) onto the cached HuggingFace snapshot so ``trust_remote_code``
loads the fix the upstream README asks users to apply by hand.
"""
import os
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parent
ENV_FILE = REPO / ".env"
MPLUG_PATCH = REPO / "mplug_owl3" / "modeling_mplugowl3.py"
MPLUG_REMOTE_FILE = "modeling_mplugowl3.py"


def _parse_env_file(env_file: Path) -> dict[str, str]:
    if not env_file.is_file():
        return {}
    values: dict[str, str] = {}
    for raw_line in env_file.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def load_env(env_file: Path = ENV_FILE) -> None:
    for key, value in _parse_env_file(env_file).items():
        os.environ.setdefault(key, value)

    token = os.environ.get("HF_TOKEN")
    if token:
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)

    cache = os.environ.get("HF_CACHE")
    if cache:
        os.environ.setdefault("HF_HOME", cache)


def _runtime_choice(env_key: str, config_attr: str, default: str) -> str:
    override = os.environ.get(env_key)
    if override:
        return override.lower()
    try:
        import config

        return str(getattr(config, config_attr, default)).lower()
    except ImportError:
        return default


def greedy_generation() -> bool:
    """Whether to decode answers/explanations greedily instead of sampling.

    Paper behavior samples (``do_sample=True``), which is irreproducible run to run.
    Greedy decoding makes the generated text -- and thus the CC-SHAP / T-SHAP scores
    computed on it -- comparable across runs. Selected via ``--decoding`` (or ``config.DECODING``).
    """
    return _runtime_choice("CCVLM_DECODING", "DECODING", "sample") == "greedy"


def model_dtype():
    """Torch dtype for model weights and image inputs.

    Paper default is fp16. fp16 matmul rounding is non-associative and, amplified across
    the SHAP partition explainer's many masked forward passes, shifts the approximate SHAP
    ratios run to run; fp32 removes that noise for deterministic, comparable scores.
    Selected via ``--dtype`` (or ``config.DTYPE``).
    """
    import torch

    choice = _runtime_choice("CCVLM_DTYPE", "DTYPE", "float16")
    return {"float32": torch.float32, "fp32": torch.float32}.get(choice, torch.float16)


def maybe_seed() -> Optional[int]:
    raw = os.environ.get("CCVLM_SEED")
    if not raw:
        return None
    seed = int(raw)

    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed


def _overlay(dest: Path, patched: bytes) -> bool:
    if dest.is_file() and dest.read_bytes() == patched:
        return False
    if dest.is_symlink():
        dest.unlink()  # replace the cache symlink with a real file, don't edit the blob
    dest.write_bytes(patched)
    return True


def ensure_mplug_remote_code(model_id: str, revision: Optional[str] = None) -> None:
    """Overlay the vendored patched ``modeling_mplugowl3.py`` onto the HF cache.

    ``trust_remote_code`` copies the model's ``.py`` files from the snapshot into
    ``$HF_HOME/modules`` and imports them, so we patch both the snapshot source (what
    transformers copies from) and any already-materialized module copy.
    """
    from huggingface_hub import hf_hub_download

    patched = MPLUG_PATCH.read_bytes()
    snapshot_file = Path(
        hf_hub_download(model_id, MPLUG_REMOTE_FILE, revision=revision)
    )
    _overlay(snapshot_file, patched)

    modules_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    for module_copy in (modules_root / "modules").rglob(MPLUG_REMOTE_FILE):
        _overlay(module_copy, patched)
