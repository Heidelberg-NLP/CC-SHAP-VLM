"""VALSE instrument sweeps that complement the per-task faithfulness run.

Three evaluations from the paper that ``evaluation.run`` does not cover:

* ``run_accuracy``     -- pairwise accuracy (``acc_r``) plus caption-only / foil-only
                          precision (``p_c`` / ``p_f``), over every configured instrument.
* ``run_mm_shap``      -- pairwise accuracy plus MM-SHAP (modality importance), same sweep.
* ``run_non_pairwise`` -- caption-only and foil-only precision + MM-SHAP for one instrument,
                          the non-pairwise framing the pairwise runs never exercise.

MM-SHAP is the text share of attribution returned by ``cc_shap.explain``.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from . import cc_shap
from .config import LABELS, MULT_CHOICE_DATA
from .data import prepare_valse_samples
from .matching import evaluate_prediction
from .models import VLMAdapter

RESULTS_DIR = Path("results_json_valse")
_BINARY = LABELS["binary"]


def run_accuracy(adapter: VLMAdapter, model_name: str, data_root: str,
                 write: bool = True) -> None:
    for task in _instruments():
        totals = {"acc_r": 0, "p_c": 0, "p_f": 0}
        records: dict = {}
        samples = prepare_valse_samples(task, data_root)
        for k, sample in enumerate(tqdm(samples)):
            image = Image.open(sample.image_path).convert("RGB")
            per_sample = {
                "acc_r": _accuracy(adapter, sample.pairwise, sample.correct_answer,
                                   task, image),
                "p_c": _accuracy(adapter, sample.caption_only, "A", task, image),
                "p_f": _accuracy(adapter, sample.foil_only, "B", task, image),
            }
            for key, value in per_sample.items():
                totals[key] += value
            records[f"{task}_{model_name}_{k}"] = {
                "image_path": sample.image_path,
                "sample": sample.pairwise,
                "correct_answer": sample.correct_answer,
                "post-hoc": per_sample,
            }
        if write and samples:
            _write(task, model_name, len(samples), records)
        _report(task, model_name, len(samples),
                {"acc_r %": totals["acc_r"], "p_c %": totals["p_c"],
                 "p_f %": totals["p_f"]})


def run_mm_shap(adapter: VLMAdapter, model_name: str, data_root: str,
                num_samples: int | None = None, write: bool = True) -> None:
    for task in _instruments():
        acc_r, t_shap_r = 0, 0.0
        records: dict = {}
        samples = prepare_valse_samples(task, data_root, num_samples)
        for k, sample in enumerate(tqdm(samples)):
            image = Image.open(sample.image_path).convert("RGB")
            ask = adapter.build_answer_input(sample.pairwise, task)
            acc_r_sample = _correct(adapter, ask, sample.correct_answer, task, image)
            acc_r += acc_r_sample
            t_shap_r += _mm_score(adapter, ask, image)
            records[f"{task}_{model_name}_{k}"] = {
                "image_path": sample.image_path,
                "sample": sample.pairwise,
                "correct_answer": sample.correct_answer,
                "post-hoc": {"acc_r": acc_r_sample},
            }
        if write and samples:
            _write(task, model_name, len(samples), records)
        _report(task, model_name, len(samples),
                {"acc_r %": acc_r, "T-SHAP_r %": t_shap_r})


def run_non_pairwise(adapter: VLMAdapter, task: str, model_name: str, data_root: str,
                     num_samples: int | None = None) -> None:
    p_c, p_f, t_shap_c, t_shap_f = 0, 0, 0.0, 0.0
    samples = prepare_valse_samples(task, data_root, num_samples)
    for sample in tqdm(samples):
        image = Image.open(sample.image_path).convert("RGB")
        ask_caption = adapter.build_answer_input(sample.caption_only, task)
        p_c += _correct(adapter, ask_caption, "A", task, image)
        t_shap_c += _mm_score(adapter, ask_caption, image)
        ask_foil = adapter.build_answer_input(sample.foil_only, task)
        p_f += _correct(adapter, ask_foil, "B", task, image)
        t_shap_f += _mm_score(adapter, ask_foil, image)
    _report(task, model_name, len(samples),
            {"p_c %": p_c, "p_f %": p_f, "T-SHAP_c %": t_shap_c, "T-SHAP_f %": t_shap_f})


def _instruments() -> list[str]:
    return [task for task in MULT_CHOICE_DATA if task != "mscoco"]


def _accuracy(adapter: VLMAdapter, formatted: str, expected: str, task: str, image) -> int:
    return _correct(adapter, adapter.build_answer_input(formatted, task), expected, task,
                    image)


def _correct(adapter: VLMAdapter, ask, expected: str, task: str, image) -> int:
    return evaluate_prediction(adapter.predict(ask, image, task, _BINARY), expected, task)


def _mm_score(adapter: VLMAdapter, ask, image) -> float:
    return cc_shap.explain(adapter, ask, image, max_new_tokens=1)[1]


def _write(task: str, model_name: str, count: int, records: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{task}_{model_name}_{count}_valse_eval.json").write_text(
        json.dumps(records))


def _report(task: str, model_name: str, count: int, metrics: dict[str, float]) -> None:
    print(f"Ran valse eval on {task} {count} samples with model {model_name}.\n")
    for label, value in metrics.items():
        pct = value * 100 / count if count else 0.0
        print(f"{label:<16}: {pct:.2f}  ")
