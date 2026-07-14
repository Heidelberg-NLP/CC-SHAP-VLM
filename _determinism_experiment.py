"""One-off: does single-GPU + deterministic CUDA kernels make the SHAP scores bitwise stable?

Runs the modernized stack twice (fp16, greedy, seed 0, --deterministic) pinned to one GPU
and diffs the two result JSONs with the regression harness's field gating.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
from regression_test import compare, find_result_json  # noqa: E402

REPO = Path(__file__).resolve().parent
TASK, MODEL, NUM = "foil_it", "mplug-owl3-7b", 2


def run(tag: str) -> dict:
    for stale in (REPO / "results_json").glob(f"{TASK}_{MODEL}_*.json"):
        stale.unlink()
    cmd = ["uv", "run", "python", "run-faithfulness.py", TASK, MODEL, str(NUM), "1",
           "data/", "--decoding", "greedy", "--seed", "0", "--dtype", "float16",
           "--deterministic"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
    print(f"\n=== [{tag}] {' '.join(cmd)} (CUDA_VISIBLE_DEVICES=0) ===", flush=True)
    start = time.monotonic()
    proc = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
    print(f"[{tag}] wall-clock: {time.monotonic() - start:.1f}s")
    if proc.returncode != 0:
        print(proc.stderr[-3000:], file=sys.stderr)
        raise SystemExit(f"[{tag}] run failed (exit {proc.returncode})")
    out = find_result_json(TASK, MODEL)
    if out is None:
        raise SystemExit(f"[{tag}] produced no result json")
    return json.loads(out.read_text())


if __name__ == "__main__":
    a = run("A")
    b = run("B")
    compare(a, b)
