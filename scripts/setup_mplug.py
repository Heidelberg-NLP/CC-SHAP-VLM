"""Download the mPLUG-Owl3 snapshot and apply the vendored ``_decode`` patch.

The patched ``modeling_mplugowl3.py`` (Apache-2.0) lives under ``mplug_owl3/``; this fetches
the full model snapshot (~16 GB weights + remote code) into the HuggingFace cache and
overlays the patched file so ``trust_remote_code=True`` loads the fix. ``load_models`` also
overlays automatically at load time, so this script is only a convenience for pre-fetching.

Usage (inside the uv env):
    uv run python scripts/setup_mplug.py                       # default: 7B, pinned rev
    uv run python scripts/setup_mplug.py --model mplug-owl3-2b
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hf_env import load_env

load_env()

from huggingface_hub import snapshot_download

from ccshapvlm.config import MODELS, MPLUG_REVISIONS
from ccshapvlm.models.mplug_owl3 import ensure_mplug_remote_code


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mplug-owl3-7b", choices=sorted(MPLUG_REVISIONS))
    args = ap.parse_args()

    model_id = MODELS[args.model]
    revision = MPLUG_REVISIONS[args.model]
    print(f"Downloading {model_id} (revision={revision}) ...")
    snapshot_download(model_id, revision=revision)

    ensure_mplug_remote_code(model_id, revision)
    print(f"Applied patched modeling_mplugowl3.py for {model_id}.")


if __name__ == "__main__":
    main()
