"""Before/after regression harness for CC-SHAP-VLM.

Runs the same fixed foil_it sample through both dependency stacks:
  * "before" : the original Python 3.12 pip stack (`ccshap-vlm-before`, via micromamba)
  * "after"  : the modernized Python 3.12 uv env (`.venv`)

Both stacks share the *same* vendored `shap/` source and the same patched mPLUG-Owl3 code,
so this guards the modernization (uv management + code fixes) against behavior changes.
Both stacks run with ``--decoding greedy --seed 0`` (and optionally ``--dtype float32``) so
the generated explanations -- and thus the CC-SHAP scores computed on them -- are comparable.

The "before" stack is slow and rarely changes, so its output is recorded on disk under
regression_results/before/ and reused: run once with ``--env before``, then ``--env after``
compares the current code against the recorded baseline without re-running it.

Wall-clock time per stack is logged (regression_results/<stack>/<key>.time) to inform the
speed optimizations planned as a last step.

Verdict: the deterministic invariants (predictions/labels, accuracy, and the discrete
faithfulness tests) must match exactly; the continuous CC-SHAP / T-SHAP scores are gated
with a wide, GPU-noise-floor tolerance (see the ``field_tolerance`` notes) because they are
not bitwise reproducible on GPU even within a single stack.

Usage:
    uv run python scripts/regression_test.py --model mplug-owl3-7b --num 2 --env before
    uv run python scripts/regression_test.py --model mplug-owl3-7b --num 2  # vs recorded
    # optionally loosen/tighten the SHAP tolerances or run in full precision:
    uv run python scripts/regression_test.py --num 2 --dtype float32 --t-shap-tol 8
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MICROMAMBA = REPO / "bin" / "micromamba"
MAMBA_ROOT = REPO / ".micromamba"
BEFORE_ENV = "ccshap-vlm-before"
RESULTS = REPO / "regression_results"
DATA_ROOT = "data/"

SKIP_SUBTREES = {"shap_plot_info_post_hoc", "shap_plot_info_cot"}


# The verdict gates strictly on the *deterministic invariants* -- the discrete outcomes that
# are reproducible run to run: predictions/labels, accuracy, and the discrete faithfulness
# tests (atanasova/turpin/lanham). Those must match exactly.
#
# The continuous CC-SHAP / T-SHAP scores are NOT bitwise reproducible on GPU: a run does ~600
# masked fp16 forward passes through SDPA attention across two devices (device_map="auto"),
# whose non-associative/atomic reductions torch cannot fully pin down, and the Permutation
# explainer amplifies those tiny differences into different ratios. Measured run-to-run noise
# floor (SAME code + seed, uv vs uv): cc_shap up to ~0.25, t-shap up to ~6 points -- on par
# with the before/after gap, i.e. the modernization adds no drift beyond this floor. So the
# SHAP scores are gated with a wide, noise-floor-derived tolerance (catches real regressions,
# tolerates the GPU noise). The `other_measures_*` distances stay purely informational.
CC_SHAP_TOL = 0.35   # cc_shap-posthoc / cc_shap-cot, range ~[-1, 1]
T_SHAP_TOL = 10.0    # t-shap_* image/text contribution percentages, range [0, 100]


def field_tolerance(field: str) -> float | None:
    """Gate tolerance for a numeric field; ``None`` means informational (never gated)."""
    if "other_measures" in field:
        return None
    if "cc_shap" in field:
        return CC_SHAP_TOL
    if "t-shap" in field:
        return T_SHAP_TOL
    return 0.0  # deterministic invariant: accuracy / atanasova / turpin / lanham


def key(task: str, model: str) -> str:
    return f"{task}_{model}"


def find_result_json(task: str, model: str) -> Path | None:
    finals = sorted((REPO / "results_json").glob(f"{task}_{model}_*_final.json"),
                    key=lambda p: p.stat().st_mtime)
    return finals[-1] if finals else None


def saved_paths(stack: str, task: str, model: str) -> tuple[Path, Path]:
    base = RESULTS / stack / key(task, model)
    return base.with_suffix(".json"), base.with_suffix(".time")


def run_stack(stack: str, task: str, model: str, num: int,
              seed: int, dtype: str | None = None) -> tuple[dict, float]:
    args = ["run-faithfulness.py", task, model, str(num), "1", DATA_ROOT,
            "--decoding", "greedy", "--seed", str(seed)]
    if dtype:
        args += ["--dtype", dtype]
    env = dict(os.environ)
    if stack == "before":
        env["MAMBA_ROOT_PREFIX"] = str(MAMBA_ROOT)
        cmd = [str(MICROMAMBA), "run", "-n", BEFORE_ENV, "python", *args]
    else:
        cmd = ["uv", "run", "python", *args]

    for stale in (REPO / "results_json").glob(f"{task}_{model}_*.json"):
        stale.unlink()

    print(f"\n=== [{stack}] {' '.join(cmd)} ===", flush=True)
    start = time.monotonic()
    proc = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
    elapsed = time.monotonic() - start

    summary = [ln for ln in proc.stdout.splitlines() if "%" in ln or "mean score" in ln]
    print("\n".join(summary) if summary else proc.stdout[-500:])
    print(f"[{stack}] wall-clock: {elapsed:.1f}s")
    if proc.returncode != 0:
        print(f"[{stack}] STDERR tail:\n{proc.stderr[-2000:]}", file=sys.stderr)
        raise SystemExit(f"[{stack}] run failed (exit {proc.returncode})")
    out_path = find_result_json(task, model)
    if out_path is None:
        raise SystemExit(f"[{stack}] produced no result json in results_json/")

    json_dest, time_dest = saved_paths(stack, task, model)
    json_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(out_path, json_dest)
    time_dest.write_text(f"{elapsed:.3f}\n")
    return json.loads(out_path.read_text()), elapsed


def flatten(entry: dict, prefix: str = "") -> tuple[dict[str, float], dict[str, str]]:
    numeric: dict[str, float] = {}
    categorical: dict[str, str] = {}
    for field, value in entry.items():
        if field in SKIP_SUBTREES:
            continue
        name = f"{prefix}{field}"
        if isinstance(value, dict):
            sub_num, sub_cat = flatten(value, f"{name}.")
            numeric.update(sub_num)
            categorical.update(sub_cat)
        elif isinstance(value, (int, float)):  # bool is an int subclass, handled here
            numeric[name] = float(value)
        elif isinstance(value, str):
            try:
                numeric[name] = float(value)
            except ValueError:
                categorical[name] = value
    return numeric, categorical


def collect_diffs(before: dict, after: dict, shared: list[str]):
    """Per-field absolute numeric diffs, plus categorical (prediction) agreement."""
    per_field: dict[str, list[float]] = {}
    categorical_total = categorical_mismatch = 0
    for k in shared:
        bn, bc = flatten(before[k])
        an, ac = flatten(after[k])
        for field, value in bn.items():
            if field in an:
                per_field.setdefault(field, []).append(abs(value - an[field]))
        for field, value in bc.items():
            if field in ac:
                categorical_total += 1
                categorical_mismatch += int(value != ac[field])
    return per_field, categorical_total, categorical_mismatch


def _gate_numeric_fields(
    per_field: dict[str, list[float]],
) -> tuple[list[str], list[str], float]:
    """Print each numeric field's diff and bucket its tolerance violations."""
    invariant_failures: list[str] = []
    shap_failures: list[str] = []
    worst_info = 0.0
    for field in sorted(per_field):
        diffs = per_field[field]
        mx = max(diffs)
        tol = field_tolerance(field)
        if tol is None:
            worst_info = max(worst_info, mx)
            flag = label = "info"
        else:
            flag = "OK" if mx <= tol else "!!"
            label = "exact" if tol == 0.0 else f"±{tol:g}"
            if mx > tol:
                target = shap_failures if tol > 0.0 else invariant_failures
                target.append(f"{field}: max|Δ|={mx:.3e} > {label}")
        print(f"  {flag:5s}{field:40s} n={len(diffs):3d}  "
              f"max|Δ|={mx:.3e}  mean|Δ|={sum(diffs) / len(diffs):.3e}  tol={label}")
    return invariant_failures, shap_failures, worst_info


def compare(before: dict, after: dict) -> bool:
    shared = [k for k in before if k in after]
    print("\n================ comparison ================")
    print(f"samples compared: {len(shared)}")

    per_field, cat_total, cat_mismatch = collect_diffs(before, after, shared)
    if not per_field:
        raise SystemExit("no comparable numeric fields (did both runs write results?)")

    invariant_failures, shap_failures, worst_info = _gate_numeric_fields(per_field)

    cat_ok = cat_mismatch == 0
    if cat_total:
        flag = "OK" if cat_ok else "!!"
        print(f"\n  {flag:5s}categorical (predictions/labels): "
              f"{cat_total - cat_mismatch}/{cat_total} agree  tol=exact")
    print(f"auxiliary other_measures worst max|Δ| = {worst_info:.3e} (informational)")
    print("--------------------------------------------")

    deterministic_ok = cat_ok and not invariant_failures
    passed = deterministic_ok and not shap_failures
    for line in invariant_failures:
        print(f"  DETERMINISTIC-FAIL {line}")
    if not cat_ok:
        print(f"  DETERMINISTIC-FAIL predictions differ: {cat_mismatch}/{cat_total} mismatched")
    for line in shap_failures:
        print(f"  SHAP-FAIL {line}")
    print(f"deterministic invariants: {'OK' if deterministic_ok else 'FAIL'}  |  "
          f"SHAP scores (noise-floor tol): {'OK' if not shap_failures else 'FAIL'}")
    print(f"verdict -> {'PASS' if passed else 'FAIL'}")
    return passed


def load_recorded(stack: str, task: str, model: str) -> dict:
    json_path, _ = saved_paths(stack, task, model)
    if not json_path.exists():
        raise SystemExit(f"no recorded {stack} baseline at {json_path}; "
                         f"run with --env {stack} first")
    return json.loads(json_path.read_text())


def main() -> None:
    global CC_SHAP_TOL, T_SHAP_TOL, RESULTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="foil_it")
    ap.add_argument("--model", default="mplug-owl3-7b")
    ap.add_argument("--num", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cc-shap-tol", type=float, default=CC_SHAP_TOL,
                    help="max allowed abs diff on cc_shap-* scores (GPU noise floor)")
    ap.add_argument("--t-shap-tol", type=float, default=T_SHAP_TOL,
                    help="max allowed abs diff on t-shap-* percentages (GPU noise floor)")
    ap.add_argument("--env", choices=["both", "before", "after"], default="after",
                    help="'before' records the baseline; 'after' compares against it")
    ap.add_argument("--dtype", choices=["float16", "float32"], default=None,
                    help="model precision; float32 checks parity free of fp16 noise")
    ap.add_argument("--tag", default=None,
                    help="results subdir (e.g. 'fp32') to keep runs from clobbering")
    args = ap.parse_args()

    CC_SHAP_TOL = args.cc_shap_tol
    T_SHAP_TOL = args.t_shap_tol
    if args.tag:
        RESULTS = RESULTS / args.tag

    if args.env == "before":
        run_stack("before", args.task, args.model, args.num, args.seed, args.dtype)
        return

    if args.env == "both":
        before, _ = run_stack("before", args.task, args.model, args.num, args.seed,
                              args.dtype)
    else:
        before = load_recorded("before", args.task, args.model)

    after, _ = run_stack("after", args.task, args.model, args.num, args.seed, args.dtype)
    ok = compare(before, after)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
