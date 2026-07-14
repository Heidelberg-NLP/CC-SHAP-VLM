"""Run the faithfulness suite over one task: per-sample measures, JSON record, and report.

Per sample we take the model's answer (direct and via chain of thought), its post-hoc
explanation, then the enabled faithfulness measures -- CC-SHAP (post-hoc and CoT) plus the
optional discrete consistency tests -- and aggregate to the reported percentages.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from . import cc_shap
from .cc_shap import ZERO_RESULT, CCShapResult
from .config import HELPER_MODEL, LABELS, MAX_NEW_TOKENS, TESTS, is_multiple_choice, is_open_ended
from .consistency_tests import ZERO_LANHAM, ConsistencyTests, LanhamResult
from .data import Sample, prepare_samples
from .helper_model import HelperModel
from .matching import evaluate_prediction
from .models import SamplePredictions, VLMAdapter

RESULTS_DIR = Path("results_json")
_HELPER_TESTS = ("atanasova_counterfactual", "turpin", "lanham")


@dataclass
class SampleOutput:
    predictions: SamplePredictions
    accuracy: int
    accuracy_cot: int
    atanasova: tuple
    turpin: tuple
    lanham: LanhamResult
    post_hoc: CCShapResult
    cot: CCShapResult


@dataclass
class Totals:
    accuracy: int = 0
    accuracy_cot: int = 0
    atanasova_counterfact: int = 0
    turpin: int = 0
    lanham_early: int = 0
    lanham_mistake: int = 0
    lanham_paraphrase: int = 0
    lanham_filler: int = 0
    cc_shap_post_hoc: float = 0.0
    cc_shap_cot: float = 0.0
    t_shap_post_hoc: float = 0.0
    t_shap_expl_post_hoc: float = 0.0
    t_shap_cot: float = 0.0
    t_shap_expl_cot: float = 0.0


def labels_for(task: str) -> list[str] | None:
    if is_multiple_choice(task):
        return LABELS["binary"]
    if is_open_ended(task):
        return None
    return LABELS[task]


def _evaluate_sample(adapter: VLMAdapter, tests: ConsistencyTests, task: str,
                     sample: Sample, labels: list[str] | None) -> SampleOutput:
    image = Image.open(sample.image_path).convert("RGB")
    preds = adapter.run_predictions(sample.formatted_input, image, task, labels,
                                    MAX_NEW_TOKENS)

    accuracy = evaluate_prediction(preds.prediction, sample.correct_answer, task)
    accuracy_cot = evaluate_prediction(preds.prediction_cot, sample.correct_answer, task)

    atanasova = (tests.atanasova_counterfactual(sample.formatted_input, image,
                                                preds.prediction, task)
                 if "atanasova_counterfactual" in TESTS else (0, 0))
    post_hoc = (cc_shap.measure(adapter, preds.ask_for_prediction, preds.prediction,
                                preds.ask_for_post_hoc_expl, image, task,
                                expl_type="post_hoc", max_new_tokens=MAX_NEW_TOKENS)
                if "cc_shap-posthoc" in TESTS else ZERO_RESULT)
    turpin = (tests.turpin(sample.formatted_input, image, preds.prediction_cot,
                           sample.correct_answer, sample.wrong_answer, task, MAX_NEW_TOKENS)
              if "turpin" in TESTS else (0, 0))
    lanham = (tests.lanham(preds.prediction_cot, preds.generated_cot, preds.ask_for_cot,
                           image, task, MAX_NEW_TOKENS)
              if "lanham" in TESTS else ZERO_LANHAM)
    cot = (cc_shap.measure(adapter, preds.ask_for_prediction, preds.prediction,
                           preds.ask_for_cot, image, task,
                           prediction_shap=post_hoc.prediction_shap, expl_type="cot",
                           max_new_tokens=MAX_NEW_TOKENS)
           if "cc_shap-cot" in TESTS else ZERO_RESULT)

    return SampleOutput(preds, accuracy, accuracy_cot, atanasova, turpin, lanham,
                        post_hoc, cot)


def _accumulate(totals: Totals, out: SampleOutput) -> None:
    totals.accuracy += out.accuracy
    totals.accuracy_cot += out.accuracy_cot
    totals.atanasova_counterfact += out.atanasova[0]
    totals.turpin += out.turpin[0]
    totals.lanham_early += out.lanham.early_answering
    totals.lanham_mistake += out.lanham.adding_mistakes
    totals.lanham_paraphrase += out.lanham.paraphrasing
    totals.lanham_filler += out.lanham.filler_tokens
    totals.cc_shap_post_hoc += out.post_hoc.score
    totals.cc_shap_cot += out.cot.score
    totals.t_shap_post_hoc += out.post_hoc.mm_score
    totals.t_shap_expl_post_hoc += out.post_hoc.mm_score_expl
    totals.t_shap_cot += out.cot.mm_score
    totals.t_shap_expl_cot += out.cot.mm_score_expl


def _other_measures(result: CCShapResult) -> dict:
    return {"dist_correl": f"{result.dist_correl:.2f}", "mse": f"{result.mse:.2f}",
            "var": f"{result.var:.2f}", "kl_div": f"{result.kl_div:.2f}",
            "js_div": f"{result.js_div:.2f}"}


def _record(sample: Sample, out: SampleOutput) -> dict:
    preds, post_hoc, cot = out.predictions, out.post_hoc, out.cot
    return {
        "image_path": sample.image_path,
        "sample": sample.formatted_input,
        "correct_answer": sample.correct_answer,
        "post_hoc_explanation": preds.post_hoc_explanation,
        "post-hoc": {
            "prediction": preds.prediction,
            "accuracy": out.accuracy,
            "shap_plot_info_post_hoc": post_hoc.plot_info,
            "cc_shap-posthoc": f"{post_hoc.score:.2f}",
            "t-shap_post_hoc": f"{post_hoc.mm_score * 100:.0f}",
            "t-shap_expl_post_hoc": f"{post_hoc.mm_score_expl * 100:.0f}",
            "atanasova_counterfact": out.atanasova[0],
            "atanasova_counterfact_info": out.atanasova[1],
            "other_measures_post_hoc": _other_measures(post_hoc),
        },
        "cot": {
            "inp_cot_askpred": preds.ask_for_answer_after_cot,
            "pred_cot": preds.prediction_cot,
            "accuracy_cot": out.accuracy_cot,
            "shap_plot_info_cot": cot.plot_info,
            "turpin": out.turpin[0],
            "turpin_info": out.turpin[1],
            "lanham_early": out.lanham.early_answering,
            "lanham_early_info": out.lanham.info,
            "lanham_mistake": out.lanham.adding_mistakes,
            "lanham_paraphrase": out.lanham.paraphrasing,
            "lanham_filler": out.lanham.filler_tokens,
            "cc_shap-cot": f"{cot.score:.2f}",
            "t-shap_cot": f"{cot.mm_score * 100:.0f}",
            "t-shap_expl_cot": f"{cot.mm_score_expl * 100:.0f}",
            "other_measures_cot": _other_measures(cot),
        },
    }


def _report(task: str, model_name: str, totals: Totals, count: int) -> None:
    def pct(value: float) -> float:
        return value * 100 / count

    print(f"Ran {list(TESTS)} on {task} {count} samples with model {model_name}. "
          "Reporting accuracy and faithfulness percentage.\n")
    print(f"Accuracy %                       : {pct(totals.accuracy):.2f}  ")
    print(f"Atanasova Counterfact %          : {pct(totals.atanasova_counterfact):.2f}  ")
    print(f"CC-SHAP post-hoc mean score      : {totals.cc_shap_post_hoc / count:.2f}  ")
    print(f"Accuracy CoT %                   : {pct(totals.accuracy_cot):.2f}  ")
    print(f"Turpin %                         : {pct(totals.turpin):.2f}  ")
    print(f"Lanham Early Answering %         : {pct(totals.lanham_early):.2f}  ")
    print(f"Lanham Filler %                  : {pct(totals.lanham_filler):.2f}  ")
    print(f"Lanham Mistake %                 : {pct(totals.lanham_mistake):.2f}  ")
    print(f"Lanham Paraphrase %              : {pct(totals.lanham_paraphrase):.2f}  ")
    print(f"CC-SHAP CoT mean score           : {totals.cc_shap_cot / count:.2f}  ")
    print(f"T-SHAP post-hoc mean score %     : {pct(totals.t_shap_post_hoc):.2f}  ")
    print(f"T-SHAP expl post-hoc mean score %: {pct(totals.t_shap_expl_post_hoc):.2f}  ")
    print(f"T-SHAP CoT mean score %          : {pct(totals.t_shap_cot):.2f}  ")
    print(f"T-SHAP expl CoT mean score %     : {pct(totals.t_shap_expl_cot):.2f}  ")


def _write_results(task: str, model_name: str, count: int, results: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{task}_{model_name}_{count}_final.json").write_text(
        json.dumps(results))


def run(adapter: VLMAdapter, task: str, model_name: str, num_samples: int,
        data_root: str, write: bool = True) -> None:
    labels = labels_for(task)
    helper = HelperModel.load(HELPER_MODEL) if any(t in TESTS for t in _HELPER_TESTS) else None
    tests = ConsistencyTests(adapter, helper, labels or [])

    samples = prepare_samples(task, data_root, num_samples)
    print("Done preparing data. Running test...")

    totals = Totals()
    results: dict = {}
    for k, sample in enumerate(tqdm(samples)):
        out = _evaluate_sample(adapter, tests, task, sample, labels)
        _accumulate(totals, out)
        results[f"{task}_{model_name}_{k}"] = _record(sample, out)

    count = len(samples)
    if write:
        _write_results(task, model_name, count, results)
    _report(task, model_name, totals, count)
